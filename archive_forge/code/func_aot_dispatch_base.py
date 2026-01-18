import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .runtime_wrappers import (
from .schemas import (
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import (
def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext
    with context(), track_graph_compiling(aot_config, 'inference'):
        compiler = aot_config.inference_compiler if aot_config.inference_compiler is not None else aot_config.fw_compiler
        if config.functionalize_rng_ops:
            fake_mode = detect_fake_mode()
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            updated_flat_args.extend([seed, offset])
        if (tracing_context := torch._guards.TracingContext.try_get()):
            tracing_context.fw_metadata = fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        compiled_fw = compiler(fw_module, updated_flat_args)
    if not hasattr(compiled_fw, '_boxed_call'):
        compiled_fw = make_boxed_func(compiled_fw)

    @wraps(compiled_fw)
    def rng_functionalization_wrapper(args):
        if fw_metadata.is_rng_op_functionalized:
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
            args.extend([seed, offset])
            out = compiled_fw(args)
            out = functionalized_rng_runtime_epilogue(fw_metadata, out)
            return out
        else:
            return compiled_fw(args)
    if maybe_subclass_meta is not None:
        compiled_fw_func = aot_dispatch_subclass_wrapper(rng_functionalization_wrapper, subclass_metas=fw_metadata.subclass_fw_graph_out_meta, num_fw_outs_saved_for_bw=None)
    else:
        compiled_fw_func = rng_functionalization_wrapper
    if not hasattr(compiled_fw_func, '_boxed_call'):
        compiled_fw_func = make_boxed_func(compiled_fw_func)
    compiled_fn = create_runtime_wrapper(compiled_fw_func, runtime_metadata=fw_metadata, indices_of_inps_to_detach=[], trace_joint=False, keep_input_mutations=aot_config.keep_inference_input_mutations, disable_amp=disable_amp)
    return compiled_fn