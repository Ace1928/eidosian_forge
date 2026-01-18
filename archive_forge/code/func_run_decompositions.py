import copy
import dataclasses
import functools
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import (  # noqa: F401
@_disable_prexisiting_fake_mode
def run_decompositions(self, decomp_table: Optional[Dict[torch._ops.OperatorBase, Callable]]=None) -> 'ExportedProgram':
    """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.
        """
    from torch._decomp import core_aten_decompositions
    from torch._export.passes.add_runtime_assertions_for_constraints_pass import _AddRuntimeAssertionsForInlineConstraintsPass, InputDim
    from torch._export.passes.lift_constant_tensor_pass import lift_constant_tensor_pass
    from torch._export.passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
    from torch._functorch.aot_autograd import aot_export_module

    def _get_placeholders(gm):
        placeholders = []
        for node in gm.graph.nodes:
            if node.op != 'placeholder':
                break
            placeholders.append(node)
        return placeholders
    decomp_table = decomp_table or core_aten_decompositions()
    old_placeholders = _get_placeholders(self.graph_module)
    fake_args = [node.meta['val'] for node in old_placeholders]
    buffers_to_remove = [name for name, _ in self.graph_module.named_buffers()]
    for name in buffers_to_remove:
        delattr(self.graph_module, name)
    gm, graph_signature = aot_export_module(self.graph_module, fake_args, decompositions=decomp_table, trace_joint=False)

    def update_arg(old_arg, new_ph):
        if isinstance(old_arg, ConstantArgument):
            return old_arg
        elif isinstance(old_arg, TensorArgument):
            return TensorArgument(name=new_ph.name)
        elif isinstance(old_arg, SymIntArgument):
            return SymIntArgument(name=new_ph.name)
        raise RuntimeError(f'Type of old_arg not supported: {type(old_arg)}')
    new_placeholders = _get_placeholders(gm)
    new_outputs = list(gm.graph.nodes)[-1].args[0]
    input_specs = [InputSpec(spec.kind, update_arg(spec.arg, new_placeholders[i]), spec.target) for i, spec in enumerate(self.graph_signature.input_specs)]
    output_specs = [OutputSpec(spec.kind, update_arg(spec.arg, new_outputs[i]), spec.target) for i, spec in enumerate(self.graph_signature.output_specs)]
    assert len(new_placeholders) == len(old_placeholders)
    old_new_placeholder_map = {old_node.name: new_node.name for old_node, new_node in zip(old_placeholders, new_placeholders)}
    new_graph_signature = ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
    for old_node, new_node in zip(old_placeholders, new_placeholders):
        if not isinstance(old_node.meta['val'], torch.Tensor):
            new_node.meta['val'] = old_node.meta['val']
        if new_node.target in new_graph_signature.inputs_to_parameters or new_node.target in new_graph_signature.inputs_to_buffers:
            for k, v in old_node.meta.items():
                new_node.meta[k] = v
    gm.meta.update(self.graph_module.meta)
    new_range_constraints = _get_updated_range_constraints(gm)
    new_equality_constraints = [(InputDim(old_new_placeholder_map[inp_dim1.input_name], inp_dim1.dim), InputDim(old_new_placeholder_map[inp_dim2.input_name], inp_dim2.dim)) for inp_dim1, inp_dim2 in self.equality_constraints]
    lift_constant_tensor_pass(gm, new_graph_signature)
    _replace_sym_size_ops_pass(gm)
    exported_program = ExportedProgram(gm, gm.graph, new_graph_signature, self.state_dict, new_range_constraints, new_equality_constraints, copy.deepcopy(self.module_call_graph), self.example_inputs, self.verifier, self.tensor_constants)
    if len(new_range_constraints) > 0 or len(new_equality_constraints) > 0:
        exported_program = exported_program._transform(_AddRuntimeAssertionsForInlineConstraintsPass(new_range_constraints, new_equality_constraints))
    return exported_program