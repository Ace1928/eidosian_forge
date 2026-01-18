import operator
import types
import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import QuantizationAnnotation
def remove_tensor_overload_for_qdq_ops(match_pattern: GraphModule) -> None:
    """ Remove .tensor overload for quantize/dequantize ops so that we can
    use the match_pattern that we get from torchdynamo export to match the output of convert_pt2e
    """
    _MAP = {torch.ops.quantized_decomposed.quantize_per_tensor.default: torch.ops.quantized_decomposed.quantize_per_tensor, torch.ops.quantized_decomposed.dequantize_per_tensor.default: torch.ops.quantized_decomposed.dequantize_per_tensor, torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor, torch.ops.quantized_decomposed.quantize_per_tensor.tensor2: torch.ops.quantized_decomposed.quantize_per_tensor, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2: torch.ops.quantized_decomposed.dequantize_per_tensor, torch.ops.quantized_decomposed.quantize_per_channel.default: torch.ops.quantized_decomposed.quantize_per_channel, torch.ops.quantized_decomposed.dequantize_per_channel.default: torch.ops.quantized_decomposed.dequantize_per_channel, torch.ops.aten.clamp.Tensor: torch.ops.aten.clamp}
    for n in match_pattern.graph.nodes:
        if n.op != 'call_function':
            continue
        if n.target in _MAP:
            n.target = _MAP[n.target]