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
def get_node_first_input_and_output_type(node: Node, gm: GraphModule, logger_cls: Callable, node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]]) -> Tuple[NodeInputOrOutputType, NodeInputOrOutputType]:
    FUNS_IO_TYPE_FP32 = node_type_to_io_type_map['funs_io_type_fp32']
    FUNS_IO_TYPE_FP16 = node_type_to_io_type_map['funs_io_type_fp16']
    FUNS_IO_TYPE_INT8 = node_type_to_io_type_map['funs_io_type_int8']
    FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['funs_io_type_fp32_or_int8']
    MODS_IO_TYPE_FP32 = node_type_to_io_type_map['mods_io_type_fp32']
    MODS_IO_TYPE_INT8 = node_type_to_io_type_map['mods_io_type_int8']
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['mods_io_type_fp32_or_int8']
    METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['meths_io_type_fp32_or_int8']
    if node.op == 'call_function':
        if node.target in FUNS_IO_TYPE_FP32:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        if node.target in FUNS_IO_TYPE_FP16:
            return (NodeInputOrOutputType.FP16, NodeInputOrOutputType.FP16)
        elif node.target in FUNS_IO_TYPE_INT8:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        elif node.target in FUNS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            _prev_node_input_type, prev_node_output_type = get_node_first_input_and_output_type(first_arg, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, prev_node_output_type)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    elif node.op == 'call_module':
        assert node.op == 'call_module'
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        is_known_fp32_or_int8_input_module = any((isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8))
        if isinstance(mod, (logger_cls, ObserverBase, FakeQuantizeBase)) or is_known_fp32_or_int8_input_module:
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            _prev_node_input_type, prev_node_output_type = get_node_first_input_and_output_type(first_arg, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, prev_node_output_type)
        is_known_fp32_input_module = any((isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32))
        is_known_int8_input_module = any((isinstance(mod, target_type) for target_type in MODS_IO_TYPE_INT8))
        if is_known_fp32_input_module:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif is_known_int8_input_module:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    elif node.op == 'call_method':
        if node.target == 'dequantize':
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            _prev_node_input_type, prev_node_output_type = get_node_first_input_and_output_type(prev_node, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, NodeInputOrOutputType.FP32)
        elif node.target == 'to':
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            _prev_node_input_type, prev_node_output_type = get_node_first_input_and_output_type(prev_node, gm, logger_cls, node_type_to_io_type_map)
            cur_node_dtype_target = get_normalized_nth_input(node, gm, 1)
            assert cur_node_dtype_target is torch.float16, f'{cur_node_dtype_target} handling needs to be added'
            return (prev_node_output_type, NodeInputOrOutputType.FP16)
        elif node.target in METHS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            _prev_node_input_type, prev_node_output_type = get_node_first_input_and_output_type(first_arg, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, prev_node_output_type)
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    else:
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)