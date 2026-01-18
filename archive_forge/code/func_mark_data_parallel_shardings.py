import operator
from contextlib import contextmanager
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
import torch.library
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def mark_data_parallel_shardings(train_step_graph: GraphModule, num_parameters: int, num_states: int, dp_strategy_map: Dict[fx.Node, StrategyType], parallel_mode: DataParallelStyle=DataParallelStyle.FULLY_SHARD) -> None:
    """Mark the sharding for the nodes in the train_step_graph."""
    activation_idx = num_parameters + num_states
    placeholder_idx = 0
    for node in train_step_graph.graph.nodes:
        node_strategy = dp_strategy_map[node]
        if node.op == 'placeholder':
            assert isinstance(node_strategy, DataParallelStrategy)
            node_type = node_strategy.node_type
            node_strategies = node_strategy.strategies
            if node_type == NodeType.NON_TENSOR:
                node_sharding = None
            elif placeholder_idx < activation_idx:
                assert len(node_strategies) > 0, 'node_strategies should not be empty'
                if parallel_mode == DataParallelStyle.REPLICATE:
                    node_sharding = node_strategies[0]
                elif parallel_mode == DataParallelStyle.FULLY_SHARD:
                    if len(node_strategies) == 1:
                        node_sharding = node_strategies[0]
                    else:
                        node_sharding = node_strategies[1]
                elif parallel_mode == DataParallelStyle.DEFAULT:
                    raise NotImplementedError('default mode not implemented')
            else:
                assert len(node_strategies) > 0, 'node_strategies should not be empty'
                node_sharding = node_strategies[0]
            node.meta['sharding'] = node_sharding
            placeholder_idx += 1
        elif node.op == 'call_function':
            if isinstance(node_strategy, TupleStrategy):
                first_strategy = cast(DataParallelStrategy, node_strategy.childs[0])
                for child_strategy in node_strategy.childs:
                    assert isinstance(child_strategy, DataParallelStrategy)
                    assert child_strategy.strategies == first_strategy.strategies
                node_strategies = first_strategy.strategies
            else:
                assert isinstance(node_strategy, DataParallelStrategy)
                node_strategies = node_strategy.strategies
            assert len(node_strategies) <= 2, 'data parallel should have at most 2 strategies'
            if len(node_strategies) == 1:
                node.meta['sharding'] = node_strategies[0]
            elif len(node_strategies) == 2:
                if parallel_mode == DataParallelStyle.REPLICATE:
                    node.meta['sharding'] = node_strategies[0]
                elif parallel_mode == DataParallelStyle.FULLY_SHARD:
                    node.meta['sharding'] = node_strategies[1]
                else:
                    raise RuntimeError('default mode not supported yet!')
            else:
                raise RuntimeError(f'node {node} strategy length {len(node_strategies)} is not expected!')
        elif node.op == 'output':
            assert isinstance(node_strategy, DataParallelStrategy) and node_strategy.node_type == NodeType.NON_TENSOR, 'output node should not be tensor'
            node.meta['sharding'] = None
        else:
            raise RuntimeError(f'op code {node.op} not supported')