from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule
from torch.quantization.fx.graph_module import GraphModule, ObservedGraphModule
from torch.quantization.quantize_fx import Scope, ScopeContextManager
from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx
from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx
from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx
from transformers import PreTrainedModel
from transformers.utils.fx import HFTracer, check_if_model_is_supported, get_concrete_args, symbolic_trace
from ..utils import check_if_available
def specialized_quantization_tracer_creator(concrete_args: Dict[str, Any]) -> Type:
    """Creates a QuantizationTracer-like class specifying concrete_args as a class attribute."""
    return type('QuantizationTracer', (QuantizationTracer,), {'specialized_concrete_args': concrete_args})