from typing import Callable, Dict, List, Set
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
def compute_batch_dim(self, node: fx.Node, full_reduction=False) -> int:
    """Compute the batch dimension for the `node`."""
    assert self.batch_dim_size != -1, 'batch dim size is not initialized!'
    if node in self.batch_dim_map:
        return self.batch_dim_map[node]
    if node.target in self.dim_rule_map:
        view_op_rule = view_op_rules[self.dim_rule_map[node.target]]
        args_val = pytree.tree_map_only(fx.Node, lambda n: n.meta['val'], node.args)
        kwargs_val = pytree.tree_map_only(fx.Node, lambda n: n.meta['val'], node.kwargs)
        output_dim_rules = view_op_rule.dim_map(*args_val, **kwargs_val)

        def collect_input_dim(cmd: DimSpec, input_dims: Set[int]):
            if isinstance(cmd, InputDim):
                input_dims.add(cmd.input_dim)
            for inp in cmd.inputs():
                collect_input_dim(inp, input_dims)
        output_dim_to_input_dims: List[Set[int]] = []
        for inp in output_dim_rules:
            input_dims: Set[int] = set()
            collect_input_dim(inp, input_dims=input_dims)
            output_dim_to_input_dims.append(input_dims)
        operand = node.all_input_nodes[0]
        operand_batch_dim = self.get_batch_dim(operand)
        for output_dim, input_dims in enumerate(output_dim_to_input_dims):
            if operand_batch_dim in input_dims:
                self.set_batch_dim(node, output_dim)
                self.batch_dim_size = node.meta['val'].shape[output_dim]
                return output_dim
    node_val = node.meta['val']
    if isinstance(node_val, (list, tuple)):
        shapes = [val.shape for val in node_val]
    else:
        shapes = [node_val.shape]
    full_reduction = False
    for shape in shapes:
        if len(shape) == 0:
            full_reduction = True
        for i, dim_size in enumerate(shape):
            if dim_size == self.batch_dim_size:
                self.set_batch_dim(node, i)
                return i
    operands = node.all_input_nodes
    if not operands:
        self.set_batch_dim(node, -1)
        return -1
    else:
        operand_batch_dim = -1
        for operand in operands:
            if operand in self.batch_dim_map:
                operand_batch_dim = self.get_batch_dim(operand)
        if operand_batch_dim < 0:
            self.set_batch_dim(node, operand_batch_dim)
            return operand_batch_dim
        elif full_reduction:
            self.set_batch_dim(node, operand_batch_dim)
            return operand_batch_dim
        else:
            self.set_batch_dim(node, -2)
            return -2