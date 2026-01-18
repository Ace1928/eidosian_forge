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
def aot_wrapper_synthetic_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta, needs_autograd: bool, compiler_fn):
    is_inference = not needs_autograd
    flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(flat_args, fw_metadata.input_info, is_inference=is_inference)
    if synthetic_base_info is None:
        return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
    if requires_subclass_dispatch(flat_args, fw_metadata):
        raise RuntimeError('Encountered aliased inputs that are mutated in the graph, but at least one input/output\nto the graph is a tensor subclass. This is not supported today. You can try to\nremove the aliasing yourself as a workaround, or otherwise file an issue on github.')
    if aot_config.is_export:
        raise RuntimeError(f'Encountered aliased inputs that are mutated in the graph you are trying to export.\nThis functionality is currently not supported. If needed, please file a github issue.\n\nsynthetic_base_info={str(synthetic_base_info)}\n\nfw_metadata={str(fw_metadata)}\n        ')
    assert len(fw_metadata.input_info) == len(synthetic_base_info)
    fw_metadata_updated, aliased_arg_idx_with_metadata_mutations = create_synthetic_base_metadata(fw_metadata, synthetic_base_info, flat_args, flat_args_with_synthetic_bases)
    num_aliased_args_with_metadata_mutations = len(aliased_arg_idx_with_metadata_mutations)

    def _unpack_synthetic_bases(primals: Tuple[Any, ...]) -> List[Any]:
        f_args_inner = []
        for inner_idx_or_tuple in synthetic_base_info:
            if isinstance(inner_idx_or_tuple, int):
                f_args_inner.append(primals[inner_idx_or_tuple])
            else:
                inner_base_idx, view_tensor = inner_idx_or_tuple
                base = primals[inner_base_idx]
                view_arg = gen_alias_from_base(base, view_tensor, view_tensor.requires_grad)
                f_args_inner.append(view_arg)
        return f_args_inner

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        unpacked_args = _unpack_synthetic_bases(args)
        aliased_args_with_metadata_mutations = [x for i, x in enumerate(unpacked_args) if i in aliased_arg_idx_with_metadata_mutations]
        if len(aliased_args_with_metadata_mutations) > 0:
            return (*flat_fn(*unpacked_args), *aliased_args_with_metadata_mutations)
        else:
            return flat_fn(*unpacked_args)
    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(wrapped_flat_fn, keep_input_mutations=fw_metadata.keep_input_mutations, is_train=fw_metadata.is_train)(*flat_args_with_synthetic_bases)
        assert ref_fw_metadata == fw_metadata_updated, f'ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, \nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}'
    compiled_fn = compiler_fn(wrapped_flat_fn, flat_args_with_synthetic_bases, aot_config, fw_metadata=fw_metadata_updated)
    if not hasattr(compiled_fn, '_boxed_call'):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(args, fw_metadata.input_info, is_inference=is_inference)
        assert synthetic_base_info is not None
        aliased_args_w_metadata_mutations = [args[i] for i in aliased_arg_idx_with_metadata_mutations]
        args.clear()
        outs = compiled_fn(args_with_synthetic_bases)
        if num_aliased_args_with_metadata_mutations > 0:
            mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
            user_outs = outs[:-num_aliased_args_with_metadata_mutations]
            for inp, mutated_inp in zip(aliased_args_w_metadata_mutations, mutated_metadata_inps):
                inp.as_strided_(mutated_inp.size(), mutated_inp.stride(), mutated_inp.storage_offset())
            return user_outs
        return outs
    return wrapped_compiled_fn