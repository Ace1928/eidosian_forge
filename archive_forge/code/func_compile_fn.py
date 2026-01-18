import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
        Implements graph splitting, first determining a set of of buckets by counting
        parameter sizes in reverse graph order, then invoking the user/backend compiler
        to compile each subgraph. Finally, stiches compiled graphs into one graphmodule
        and returns its callable.
        """
    assert torch._inductor.config.keep_output_stride, 'Detected that you are running DDP with torch.compile, along with these two flags:\n- torch._dynamo.config.optimize_ddp = True\n- torch._inductor.config.keep_output_stride = False\nThis combination of flags is incompatible. Please set keep_output_stride to False,\nor file a github issue.'
    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    if has_higher_order_op(gm):
        raise NotImplementedError('DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph. Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.')
    buckets = [Bucket()]
    for node in reversed(gm.graph.nodes):
        if node.op in ('output', 'placeholder'):
            continue
        if buckets[0].size >= self.bucket_bytes_cap or (len(buckets) == 1 and buckets[0].size >= self.first_bucket_cap):
            if bucket_has_external_output(buckets[0]):
                buckets.insert(0, Bucket())
            else:
                if buckets[0].opcount_increased_to_capture_external_output == 0:
                    buckets[0].paramsize_before_opcount_increase = buckets[0].size
                buckets[0].opcount_increased_to_capture_external_output += 1
        if node.op == 'call_module':
            target = gm.get_submodule(node.target)
            for name, param in target.named_parameters():
                if param.requires_grad and (not self._ignore_parameter(param)):
                    buckets[0].size += param.untyped_storage().nbytes()
                    buckets[0].params.append(f'{node.target}_{name}')
                    buckets[0].param_ids.append(id(param))
        elif node.op == 'get_attr':
            maybe_param = getattr(gm, node.target)
            if maybe_param.requires_grad and (not self._ignore_parameter(maybe_param)):
                buckets[0].size += maybe_param.untyped_storage().nbytes()
                buckets[0].params.append(node.target)
                buckets[0].param_ids.append(id(maybe_param))
        buckets[0].nodes.append(node)
    if len(buckets) > 1 and buckets[0].size == 0:
        buckets[1].nodes.extend(buckets[0].nodes)
        assert len(buckets[0].params) == 0, 'Params should be empty if size is 0'
        del buckets[0]
    self.buckets = buckets
    pretty_print_buckets(buckets, self.bucket_bytes_cap)
    if len(buckets) == 1:
        return self.backend_compile_fn(gm, example_inputs)
    partition_map = {}
    for idx, b in enumerate(buckets):
        for node in b.nodes:
            partition_map[node] = idx
    split_gm = fx.passes.split_module.split_module(gm, None, lambda node: partition_map[node])
    debug_str = f'\n---orig graph---\n{gm.graph}\n' + f'\n---split graph---\n{split_gm.graph}\n'
    for name, module in split_gm.named_modules():
        if '.' not in name and len(name):
            debug_str += f'\n---{name} graph---\n{module.graph}\n'
    debug_str += '\n---------------\n'
    ddp_graph_log.debug(debug_str)

    class SubmodCompiler(torch.fx.interpreter.Interpreter):

        def __init__(self, module, compiler):
            super().__init__(module)
            self.compiler = compiler

        def compile_submod(self, input_mod, args, kwargs):
            """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
            assert len(kwargs) == 0, 'We assume only args for these modules'

            class WrapperModule(torch.nn.Module):

                def __init__(self, submod, unwrap_singleton_tuple):
                    super().__init__()
                    self.submod = submod
                    self.unwrap_singleton_tuple = unwrap_singleton_tuple

                def forward(self, *args):
                    x = self.submod(*args)
                    if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                        return x[0]
                    return x
            unwrap_singleton_tuple = False
            for sn in input_mod.graph.nodes:
                if sn.op == 'output':
                    if not isinstance(sn.args[0], tuple):
                        unwrap_singleton_tuple = True
                        sn.args = (sn.args,)
            input_mod.recompile()
            input_mod.compile_subgraph_reason = GraphCompileReason('DDPOptimizer intentional graph-break (See Note [DDPOptimizer]). Set `torch._dynamo.config.optimize_ddp = False` to disable.', [traceback.FrameSummary(__file__, 0, DDPOptimizer)])
            wrapper = WrapperModule(self.compiler(input_mod, args), unwrap_singleton_tuple)
            return wrapper

        def run_node(self, n: Node) -> Any:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            new_args = []
            assert fake_mode
            for arg in args:
                if isinstance(arg, torch.Tensor) and (not isinstance(arg, torch._subclasses.FakeTensor)):
                    new_args.append(torch._dynamo.utils.to_fake_tensor(arg, fake_mode))
                else:
                    new_args.append(arg)
            log.debug('run_node %s, %s got args %s', n.op, n.target, args_str(args))
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            if n.op == 'call_module':
                real_mod = self.fetch_attr(n.target)
                if fake_mode:
                    curr_submod = deepcopy_to_fake_tensor(real_mod, fake_mode)
                else:
                    curr_submod = real_mod
                ddp_graph_log.debug('\n---%s graph---\n%s', n.target, curr_submod.graph)
                compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
                self.module.delete_submodule(n.target)
                n.target = 'compiled_' + n.target
                self.module.add_submodule(n.target, compiled_submod_real)
                with fake_mode:
                    return curr_submod(*new_args, **kwargs)
            else:
                return getattr(self, n.op)(n.target, new_args, kwargs)
    submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn)
    submod_compiler.run(*example_inputs)
    split_gm.recompile()
    ddp_graph_log.debug('\n---final graph---\n%s\n---------------\n', split_gm.graph)
    return split_gm