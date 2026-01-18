import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
def get_node_input_qparams(node: Node, gm: GraphModule, node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]]) -> Optional[Tuple[Union[torch.Tensor, float], Union[torch.Tensor, int]]]:
    """
    Returns the qparams (scale, zero_point) of the first input to `node`,
    if they can be inferred from the graph.
    """
    prev_node = get_normalized_nth_input(node, gm, 0)
    if not isinstance(prev_node, Node):
        return None
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['mods_io_type_fp32_or_int8']

    def _get_scale_zp_from_function_args(node, gm, scale_arg_idx, zp_arg_idx):
        scale_node = get_normalized_nth_input(node, gm, scale_arg_idx)
        zp_node = get_normalized_nth_input(node, gm, zp_arg_idx)
        assert isinstance(scale_node, Node) and isinstance(scale_node.target, str)
        assert isinstance(zp_node, Node) and isinstance(zp_node.target, str)
        scale_obj = getattr_from_fqn(gm, scale_node.target)
        zp_obj = getattr_from_fqn(gm, zp_node.target)
        return (scale_obj, zp_obj)
    if prev_node.op == 'call_function':
        if prev_node.target == torch.quantize_per_tensor:
            return _get_scale_zp_from_function_args(prev_node, gm, 1, 2)
        elif prev_node.target in (toq.add, toq.add_relu, toq.mul, toq.mul_relu):
            return _get_scale_zp_from_function_args(prev_node, gm, 2, 3)
        return None
    elif prev_node.op == 'call_module':
        assert isinstance(prev_node.target, str)
        module_obj = getattr_from_fqn(gm, prev_node.target)
        if isinstance(module_obj, (nnq.Linear, nnq.Conv1d, nnq.Conv2d, nniq.ConvReLU2d, nnq.Conv3d, nnq.BatchNorm2d, nnq.BatchNorm3d, nnq.ConvTranspose1d, nnq.ConvTranspose2d, nnq.ELU, nnq.GroupNorm, nnq.InstanceNorm1d, nnq.InstanceNorm2d, nnq.InstanceNorm3d, nnq.LayerNorm, nnq.Hardswish, nnq.LeakyReLU, nnq.ReLU6, nniq.BNReLU2d, nniq.BNReLU3d, nniq.ConvReLU1d, nniq.ConvReLU2d, nniq.ConvReLU3d, nniq.LinearReLU)):
            return (module_obj.scale, module_obj.zero_point)
        is_known_fp32_or_int8_input_module = any((isinstance(module_obj, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8))
        if is_known_fp32_or_int8_input_module:
            return get_node_input_qparams(prev_node, gm, node_type_to_io_type_map)
    return None