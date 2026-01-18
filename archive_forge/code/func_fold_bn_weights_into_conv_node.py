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
def fold_bn_weights_into_conv_node(conv_node: Node, conv_weight_node: Node, conv_bias_node: Optional[Node], bn_node: Node, m: GraphModule) -> None:
    conv_w = _get_tensor_constant_from_node(conv_weight_node, m)
    conv_b = _get_tensor_constant_from_node(conv_bias_node, m)
    transpose = _is_conv_transpose(conv_node)
    bn_args_schema = bn_node.target._schema.arguments
    bn_args = _get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
    bn_w = _get_tensor_constant_from_node(bn_args[1], m)
    bn_b = _get_tensor_constant_from_node(bn_args[2], m)
    bn_rm = _get_tensor_constant_from_node(bn_args[3], m)
    bn_rv = _get_tensor_constant_from_node(bn_args[4], m)
    if bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
        eps_arg_index = 6
    elif _is_supported_batch_norm_for_training(bn_node):
        eps_arg_index = 7
    else:
        raise ValueError('BN node target is unexpected ', bn_node.target)
    bn_eps = bn_args[eps_arg_index]
    fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose)
    conv_args = list(conv_node.args)
    if len(conv_args) == 2:
        conv_args.append(None)
    weight_attr_name = conv_weight_node.target
    assert isinstance(weight_attr_name, str)
    setattr(m, weight_attr_name, fused_weight)
    if conv_bias_node is not None:
        bias_attr_name = conv_bias_node.target
        setattr(m, bias_attr_name, fused_bias)
    else:
        bias_attr_name = weight_attr_name + '_bias'
        setattr(m, bias_attr_name, fused_bias)
        with m.graph.inserting_before(conv_node):
            get_bias_node = m.graph.get_attr(bias_attr_name)
        conv_args[2] = get_bias_node
    conv_node.args = tuple(conv_args)
    for user in bn_node.users:
        if user.op != 'call_function' or user.target != operator.getitem or user.args[1] != 0:
            continue
        user.replace_all_uses_with(conv_node)