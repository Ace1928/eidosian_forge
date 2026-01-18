import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
def create_runtime_wrapper(compiled_fn, *, runtime_metadata: ViewAndMutationMeta, indices_of_inps_to_detach: List[int], trace_joint: bool, keep_input_mutations: bool, disable_amp: bool):
    if not hasattr(compiled_fn, '_boxed_call'):
        compiled_fn = make_boxed_func(compiled_fn)

    def runtime_wrapper(*args):
        if trace_joint:
            args_ = list(args)
            for idx in indices_of_inps_to_detach:
                if isinstance(args_[idx], torch.Tensor):
                    args_[idx] = args_[idx].detach()
            with torch.autograd._force_original_view_tracking(True):
                all_outs = call_func_at_runtime_with_args(compiled_fn, args_, disable_amp=disable_amp)
        else:
            with torch.no_grad():
                all_outs = call_func_at_runtime_with_args(compiled_fn, args, disable_amp=disable_amp)
        num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
        num_intermediate_bases = runtime_metadata.num_intermediate_bases
        if keep_input_mutations and trace_joint:
            num_graph_handled = runtime_metadata.num_mutated_graph_handled_indices
            if num_graph_handled > 0:
                all_outs = all_outs[:-num_graph_handled]
        assert len(all_outs) == num_mutated_runtime_inps + runtime_metadata.num_outputs + num_intermediate_bases
        num_mutations_to_apply = runtime_metadata.num_mutated_inp_runtime_indices
        if num_mutations_to_apply > 0:
            updated_inputs = all_outs[:num_mutations_to_apply]
            fw_outs = all_outs[num_mutations_to_apply:]
            for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
                meta = runtime_metadata.input_info[inpt_idx]
                if not meta.mutates_data and (not meta.mutates_metadata):
                    continue
                original_inpt = args[inpt_idx]
                updated_inpt = updated_inputs[i]
                if meta.mutates_storage_metadata:
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    with torch.no_grad():
                        original_inpt.set_(updated_inpt)
                    continue
                if meta.mutates_metadata and (not meta.mutates_data):
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    original_inpt.as_strided_(updated_inpt.size(), updated_inpt.stride(), updated_inpt.storage_offset())
                else:
                    if meta.mutates_data and meta.mutates_metadata:
                        original_inpt.as_strided_(updated_inpt.size(), updated_inpt.stride(), updated_inpt.storage_offset())
                    else:
                        assert meta.mutates_data
                    if meta.is_leaf and original_inpt.requires_grad:
                        original_inpt.detach().copy_(updated_inpt)
                    else:
                        original_inpt.copy_(updated_inpt)
        else:
            fw_outs = all_outs
        if runtime_metadata.num_outputs_aliased > 0:
            if runtime_metadata.num_intermediate_bases > 0:
                fw_outs_no_intermediate_bases = fw_outs[:-runtime_metadata.num_intermediate_bases]
                intermediate_bases = fw_outs[-runtime_metadata.num_intermediate_bases:]
            else:
                fw_outs_no_intermediate_bases = fw_outs
                intermediate_bases = []
            assert len(fw_outs_no_intermediate_bases) == len(runtime_metadata.output_info)
            fw_outs_including_aliases = []
            for i, (o, info) in enumerate(zip(fw_outs_no_intermediate_bases, runtime_metadata.output_info)):
                if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]:
                    fw_outs_including_aliases.append(o)
                    continue
                if trace_joint:
                    assert isinstance(o, TensorAlias)
                    o_ = o.alias
                else:
                    o_ = o
                o_grad = runtime_metadata.output_info[i].requires_grad
                if info.output_type == OutputType.alias_of_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.is_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = aliased_base_tensor
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.alias_of_intermediate:
                    base_tensor_list = intermediate_bases
                elif info.output_type == OutputType.alias_of_intermediate_save_as_output:
                    base_tensor_list = intermediate_bases
                else:
                    assert info.output_type == OutputType.alias_of_intermediate_base_is_user_output
                    base_tensor_list = fw_outs_no_intermediate_bases
                aliased_base_tensor = base_tensor_list[info.base_idx]
                regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                fw_outs_including_aliases.append(regenerated_out)
            ret_outs = fw_outs_including_aliases
        else:
            ret_outs = fw_outs
        if runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                if hasattr(t, '_dynamo_weak_dynamic_indices'):
                    t._dynamo_weak_dynamic_indices |= o.dynamic_dims
                else:
                    t._dynamo_weak_dynamic_indices = o.dynamic_dims.copy()
        if runtime_metadata.grad_enabled_mutation is not None:
            torch.set_grad_enabled(runtime_metadata.grad_enabled_mutation)
        return ret_outs
    return runtime_wrapper