import collections
from functools import wraps
from typing import Callable, DefaultDict, Dict, List
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .functional_utils import (
from .schemas import (
from .subclass_utils import create_subclass_meta
from .utils import _get_autocast_states, KNOWN_TYPES, strict_zip
from a multi-output view call"
def run_functionalized_fw_and_collect_metadata(f, *, keep_input_mutations: bool, is_train: bool=False, requires_subclass_dispatch: bool=False) -> Callable[..., ViewAndMutationMeta]:
    memo: Dict[Tensor, Tensor] = {}

    def _to_fun(t):
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = to_fun(t)
            memo[t] = r
            return r
        else:
            return t

    @wraps(f)
    def inner(*flat_args):
        assert all((isinstance(a, KNOWN_TYPES) for a in flat_args))
        input_info: List[InputAliasInfo] = []
        output_info: List[OutputAliasInfo] = []
        flat_f_args = pytree.tree_map(_to_fun, flat_args)
        prior_grad_enabled = torch.is_grad_enabled()
        prior_autocast_states = _get_autocast_states()
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            flat_f_outs = f(*flat_f_args)
        if prior_autocast_states != _get_autocast_states():
            raise RuntimeError('AOTAutograd does not support tracing graphs that mutate the autocast state. Dynamo will only insert autocast context managers (e.g. with torch.autocast(..)) into the graph, which will unwind all of their mutations to autocast state before the graph exits. If you encounter this error while using torch.compile, please file a bug.')
        for i, (arg, f_arg) in enumerate(zip(flat_args, flat_f_args)):
            if not isinstance(arg, Tensor):
                new_arg = arg
            else:
                new_arg = from_fun(f_arg)
            mutates_metadata = has_metadata_mutation(f_arg, arg, check_only_storage_mutation=False)
            mutates_storage_metadata = has_metadata_mutation(f_arg, arg, check_only_storage_mutation=True)
            mutates_data = has_data_mutation(f_arg)
            mutations_hidden_from_autograd = are_all_mutations_hidden_from_autograd(f_arg)
            mutations_under_no_grad_or_inference_mode = mutates_data and are_all_mutations_under_no_grad_or_inference_mode(f_arg)
            if mutates_storage_metadata:
                mutates_data = False
            requires_grad = isinstance(f_arg, torch.Tensor) and f_arg.requires_grad
            input_info.append(InputAliasInfo(is_leaf=isinstance(arg, Tensor) and safe_is_leaf(arg), mutates_data=mutates_data, mutates_metadata=mutates_metadata, mutations_hidden_from_autograd=mutations_hidden_from_autograd, mutates_storage_metadata=mutates_storage_metadata, mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode, requires_grad=requires_grad, mutation_type=_get_mutation_type(keep_input_mutations, mutates_data, mutates_metadata, mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode, requires_grad)))
        inp_storage_refs = {StorageWeakRef(inpt.untyped_storage()): idx for idx, inpt in enumerate(flat_f_args) if isinstance(inpt, Tensor)}
        inp_tensor_ids = {id(inpt) for inpt in flat_f_args if isinstance(inpt, Tensor)}
        out_tensor_ids = {id(o): i for i, o in enumerate(flat_f_outs)}
        out_tensor_alias_counts: DefaultDict = collections.defaultdict(int)
        num_aliased_tensors_that_are_multi_output_views: DefaultDict = collections.defaultdict(int)
        out_storage_to_tensors: DefaultDict = collections.defaultdict(set)
        curr_storage = None
        for o in flat_f_outs:
            if isinstance(o, torch.Tensor):
                curr_storage = StorageWeakRef(o.untyped_storage())
                out_tensor_alias_counts[curr_storage] += 1
                is_cur_tensor_multi_out_view = isinstance(o, FunctionalTensor) and torch._functionalize_is_multi_output_view(o.elem)
                if is_cur_tensor_multi_out_view:
                    num_aliased_tensors_that_are_multi_output_views[curr_storage] += 1
                out_storage_to_tensors[curr_storage].add(o)
        intermediate_base_tensor_id_to_output_idx: Dict[int, int] = {}
        intermediate_bases: List[torch.Tensor] = []
        for o in flat_f_outs:
            curr_storage = None if not isinstance(o, torch.Tensor) else StorageWeakRef(o.untyped_storage())
            outs_with_identical_metadata_that_require_grad = [] if not isinstance(o, Tensor) else [curr for curr in out_storage_to_tensors[curr_storage] if has_same_metadata(o, curr) and curr.requires_grad and (o is not curr)]
            is_result_of_custom_autograd_fn = False
            if isinstance(o, Tensor):
                if type(o.grad_fn).__name__ == 'CppFunction':
                    is_result_of_custom_autograd_fn = True
                if isinstance(o.grad_fn, torch.autograd.function.BackwardCFunction):
                    is_result_of_custom_autograd_fn = True
            if not isinstance(o, Tensor):
                output_type = OutputType.non_alias
                base_idx = None
            elif curr_storage in inp_storage_refs and o.grad_fn is not None and is_result_of_custom_autograd_fn:
                output_type = OutputType.custom_function_view
                base_idx = None
            elif curr_storage in inp_storage_refs:
                base_idx = inp_storage_refs[curr_storage]
                is_input_tensor = id(o) in inp_tensor_ids
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = num_aliased_tensors_that_are_multi_output_views[curr_storage]
                num_aliased_outs_that_are_not_multi_output_views = num_aliased_outs - num_multi_output_view_outs
                if o.grad_fn is not None and num_aliased_outs_that_are_not_multi_output_views == 0:
                    aot_graphs_log.info('Encountered AOTAutograd case: differentiable outputs that alias each other from a multi-output view call')
                    output_type = OutputType.non_alias
                elif is_input_tensor:
                    output_type = OutputType.is_input
                else:
                    output_type = OutputType.alias_of_input
            elif o._base is not None and o.requires_grad and o._base.requires_grad:
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = num_aliased_tensors_that_are_multi_output_views[curr_storage]
                num_aliased_outs_that_are_not_multi_output_views = num_aliased_outs - num_multi_output_view_outs
                if out_tensor_alias_counts[curr_storage] == 1 or num_aliased_outs_that_are_not_multi_output_views <= 1:
                    if out_tensor_alias_counts[curr_storage] != 1 and num_aliased_outs_that_are_not_multi_output_views <= 1:
                        aot_graphs_log.info('Encountered AOTAutograd case: differentiable outputs that alias each other from a multi-output view call')
                    output_type = OutputType.unsafe_view_alias
                    base_idx = None
                else:
                    maybe_existing_out_idx = out_tensor_ids.get(id(o._base), None)
                    if maybe_existing_out_idx is not None:
                        output_type = OutputType.alias_of_intermediate_base_is_user_output
                        base_idx = maybe_existing_out_idx
                    else:
                        maybe_existing_base_output_idx = intermediate_base_tensor_id_to_output_idx.get(id(o._base), None)
                        if maybe_existing_base_output_idx is not None:
                            output_type = OutputType.alias_of_intermediate
                            base_idx = maybe_existing_base_output_idx
                        else:
                            new_out_idx = len(intermediate_bases)
                            base_idx = new_out_idx
                            output_type = OutputType.alias_of_intermediate_save_as_output
                            intermediate_base_tensor_id_to_output_idx[id(o._base)] = new_out_idx
                            intermediate_bases.append(o._base)
            elif out_tensor_alias_counts[curr_storage] > 1 and len(outs_with_identical_metadata_that_require_grad) > 0 and (not o.requires_grad):
                assert len(outs_with_identical_metadata_that_require_grad) > 0
                out_alias = outs_with_identical_metadata_that_require_grad[0]
                existing_out_idx = out_tensor_ids[id(out_alias)]
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = existing_out_idx
            else:
                output_type = OutputType.non_alias
                base_idx = None
            if isinstance(o, torch.Tensor):
                dynamic_dims = {i for i, s in enumerate(o.shape) if not is_concrete_int(s)}
            else:
                dynamic_dims = None
            out_info = OutputAliasInfo(output_type=output_type, raw_type=type(o), base_idx=base_idx, dynamic_dims=dynamic_dims, requires_grad=isinstance(o, torch.Tensor) and o.requires_grad)
            output_info.append(out_info)

        def view_avoid_dupes_with_primals(t):
            if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
                return transform_subclass(t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t))
            if isinstance(t, Tensor):
                return t.view(t.shape)
            return t
        f_input_tangents = [inp for inp, info in zip(flat_f_args, input_info) if _get_mutation_type(keep_input_mutations, mutates_data=info.mutates_data, mutates_metadata=info.mutates_metadata, mutations_hidden_from_autograd=info.mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode=info.mutations_under_no_grad_or_inference_mode, requires_grad=info.requires_grad) == MutationType.MUTATED_OUT_GRAPH and info.mutates_data and info.requires_grad]
        f_output_tangents = [o for o, info in zip(flat_f_outs, output_info) if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view] and issubclass(info.raw_type, torch.Tensor) and info.requires_grad]
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases
        traced_tangents = pytree.tree_map(from_fun, f_tangents)
        traced_tangents = pytree.tree_map(view_avoid_dupes_with_primals, traced_tangents)
        user_outs = pytree.tree_map(from_fun, f_output_tangents)
        f_mutated_inputs = [inp for inp, info in zip(flat_f_args, input_info) if info.mutates_data or info.mutates_metadata]
        f_metadata_mutated_inputs = [inp for inp, info in zip(flat_f_args, input_info) if info.mutates_metadata]
        f_fw_graph_outs = list(flat_f_outs)
        if is_train or not keep_input_mutations:
            f_fw_graph_outs = f_mutated_inputs + f_fw_graph_outs
        else:
            f_fw_graph_outs = f_metadata_mutated_inputs + f_fw_graph_outs
        if is_train:
            f_fw_graph_outs = f_fw_graph_outs + intermediate_bases
        fw_graph_outs = pytree.tree_map(from_fun, f_fw_graph_outs)
        grad_enabled_mutation = None
        if torch.is_grad_enabled() != prior_grad_enabled:
            grad_enabled_mutation = torch.is_grad_enabled()
            torch.set_grad_enabled(prior_grad_enabled)
            aot_graphs_log.info('grad_mode mutation encountered in graph. Will emit mutation epilogue, to set grad_mode=%s', grad_enabled_mutation)
        metadata = ViewAndMutationMeta(input_info=input_info, output_info=output_info, num_intermediate_bases=len(intermediate_bases), keep_input_mutations=keep_input_mutations, traced_tangents=traced_tangents, subclass_inp_meta=create_subclass_meta(flat_args), subclass_fw_graph_out_meta=create_subclass_meta(fw_graph_outs), subclass_tangent_meta=create_subclass_meta(traced_tangents), is_train=is_train, grad_enabled_mutation=grad_enabled_mutation, requires_subclass_dispatch=requires_subclass_dispatch)
        return metadata
    return inner