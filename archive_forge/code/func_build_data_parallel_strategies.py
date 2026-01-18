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
def build_data_parallel_strategies(train_step_graph: GraphModule, num_params: int, num_states: int, mesh: DeviceMesh, batch_dim: int=0) -> Dict[fx.Node, StrategyType]:
    """Loop through the train step graph and build the data parallel strategy for each fx Node."""
    activation_idx = num_params + num_states
    non_compute_ops = [aten.clone.default, aten.detach.default, aten.ones_like.default, aten.reshape.default, aten.t.default, aten.view.default, torch.ops._spmd.tag_grad.default, operator.getitem]
    tuple_strategy_ops = [aten._fused_adam.default]
    dp_strategy_map: Dict[fx.Node, StrategyType] = {}
    batch_dim_analyzer = BatchDimAnalyzer(batch_dim)
    placeholder_idx = 0
    num_param_grad = 0
    for node in reversed(list(train_step_graph.graph.nodes)):
        if node.target == torch.ops._spmd.tag_grad.default:
            cur_node = node
            while cur_node.target in non_compute_ops:
                cur_node = cur_node.args[0]
                partial_strategy = _gen_partial_strategy(mesh)
                dp_strategy_map[cur_node] = DataParallelStrategy(NodeType.GRAD, [partial_strategy])
            num_param_grad += 1
            node.replace_all_uses_with(node.args[0])
            train_step_graph.graph.erase_node(node)
            if num_param_grad == num_params:
                break
    for node in train_step_graph.graph.nodes:
        if node.op == 'placeholder':
            if 'val' not in node.meta:
                dp_strategy_map[node] = DataParallelStrategy(NodeType.NON_TENSOR, [])
            elif placeholder_idx < num_params:
                shard_strategy = _gen_shard_strategy(mesh, 0)
                replica_strategy = _gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(NodeType.PARAM, [replica_strategy, shard_strategy])
            elif placeholder_idx < activation_idx:
                replica_strategy = _gen_replicate_strategy(mesh)
                shard_strategy = _gen_shard_strategy(mesh, 0)
                dp_strategy_map[node] = DataParallelStrategy(NodeType.STATE, [replica_strategy, shard_strategy])
            else:
                activation_batch_dim_size = node.meta['val'].shape[batch_dim]
                if batch_dim_analyzer.batch_dim_size == -1:
                    batch_dim_analyzer.init_batch_dim_size(activation_batch_dim_size)
                batch_dim_analyzer.set_batch_dim(node, batch_dim)
                shard_strategy = _gen_shard_strategy(mesh, batch_dim)
                dp_strategy_map[node] = DataParallelStrategy(NodeType.ACT, [shard_strategy])
            placeholder_idx += 1
        elif node.op == 'call_function':
            if node.target in non_compute_ops:
                assert node.target != torch.ops._spmd.tag_grad.default
                input_nodes = node.all_input_nodes
                assert len(input_nodes) == 1, f'non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}'
                arg_strategy = dp_strategy_map[input_nodes[0]]
                if node.target == operator.getitem:
                    getitem_idx = node.args[1]
                    if isinstance(arg_strategy, TupleStrategy):
                        dp_strategy_map[node] = arg_strategy.childs[getitem_idx]
                    else:
                        dp_strategy_map[node] = arg_strategy
                else:
                    assert isinstance(arg_strategy, DataParallelStrategy)
                    arg_node_type = arg_strategy.node_type
                    if arg_node_type == NodeType.PARAM:
                        replica_strategy = _gen_replicate_strategy(mesh)
                        dp_strategy_map[node] = DataParallelStrategy(NodeType.PARAM, [replica_strategy])
                    elif arg_node_type == NodeType.GRAD:
                        partial_sig = _gen_partial_strategy(mesh)
                        dp_strategy_map[node] = DataParallelStrategy(NodeType.GRAD, [partial_sig])
                    elif arg_node_type == NodeType.ACT:
                        arg_node_spec = batch_dim_analyzer.compute_act_spec(input_nodes[0], mesh)
                        output_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
                        shard_strategy = PlacementStrategy(output_spec=output_spec, input_specs=[arg_node_spec])
                        dp_strategy_map[node] = DataParallelStrategy(NodeType.ACT, [shard_strategy])
                    else:
                        raise RuntimeError(f'non compute op not supporting {arg_node_type}! ')
                continue
            input_args = node.all_input_nodes
            input_specs = []
            if node in dp_strategy_map:
                node_strategy = dp_strategy_map[node]
                assert isinstance(node_strategy, DataParallelStrategy)
                node_type = node_strategy.node_type
                assert node_type == NodeType.GRAD
                produce_param_grad_strat = node_strategy.strategies
                has_activation = False
                for arg in input_args:
                    arg_strategy = dp_strategy_map[arg]
                    assert isinstance(arg_strategy, DataParallelStrategy)
                    arg_node_type = arg_strategy.node_type
                    if arg_node_type == NodeType.ACT:
                        has_activation = True
                        act_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)
                        input_specs.append(act_spec)
                if has_activation:
                    assert len(produce_param_grad_strat) == 1
                    produce_param_grad_strat[0].input_specs = input_specs
            elif node.target in tuple_strategy_ops:
                output_strategy_len = len(node.args) - 1
                tuple_strategies = []
                for i in range(output_strategy_len):
                    if not isinstance(node.args[i], list):
                        raise RuntimeError(f'Expecting list as arg to build Tuple Strategy, but found type {type(node.args[i])}!')
                    if len(node.args[i]) > 0:
                        arg_strategy = dp_strategy_map[node.args[i][0]]
                        assert isinstance(arg_strategy, DataParallelStrategy)
                        assert arg_strategy.node_type in [NodeType.PARAM, NodeType.GRAD, NodeType.STATE], 'Expecting param/grad/state as arg to build Tuple Strategy!'
                        replica_strategy = _gen_replicate_strategy(mesh)
                        shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                        out_node_strategy: StrategyType = DataParallelStrategy(arg_strategy.node_type, [replica_strategy, shard_strategy])
                        tuple_strategies.append(out_node_strategy)
                output_tuple_strategy = TupleStrategy(tuple(tuple_strategies))
                dp_strategy_map[node] = output_tuple_strategy
            else:
                input_node_types = [cast(DataParallelStrategy, dp_strategy_map[arg]).node_type for arg in input_args if isinstance(dp_strategy_map[arg], DataParallelStrategy)]
                if NodeType.GRAD in input_node_types:
                    replica_strategy = _gen_replicate_strategy(mesh)
                    shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                    output_node_type = NodeType.PARAM
                    non_grad_types = [t for t in input_node_types if t != NodeType.GRAD]
                    output_node_type = non_grad_types[0]
                    for non_grad_type in non_grad_types:
                        assert non_grad_type == output_node_type, f'Found more than one non grad types! Expect {output_node_type} but found {non_grad_type}!'
                    assert output_node_type in [NodeType.PARAM, NodeType.STATE], f'Expecting output node type to be either state or param, but found {output_node_type}!'
                    dp_strategy_map[node] = DataParallelStrategy(output_node_type, [replica_strategy, shard_strategy])
                elif NodeType.STATE in input_node_types:
                    replica_strategy = _gen_replicate_strategy(mesh)
                    shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                    output_node_type = NodeType.PARAM if NodeType.PARAM in input_node_types else NodeType.STATE
                    dp_strategy_map[node] = DataParallelStrategy(output_node_type, [replica_strategy, shard_strategy])
                elif NodeType.PARAM in input_node_types:
                    if NodeType.ACT in input_node_types:
                        for arg in input_args:
                            arg_strategy = dp_strategy_map[arg]
                            assert isinstance(arg_strategy, DataParallelStrategy)
                            node_type = arg_strategy.node_type
                            if node_type == NodeType.ACT:
                                act_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)
                                input_specs.append(act_spec)
                            elif node_type == NodeType.PARAM:
                                input_specs.append(DTensorSpec(mesh=mesh, placements=(Replicate(),)))
                            else:
                                raise RuntimeError(f'Expecting node with parameter and activation, but found {input_node_types}! ')
                        output_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
                        act_strategy = PlacementStrategy(output_spec=output_spec, input_specs=input_specs)
                        dp_strategy_map[node] = DataParallelStrategy(NodeType.ACT, [act_strategy])
                    else:
                        dp_strategy_map[node] = dp_strategy_map[input_args[0]]
                else:
                    for arg in input_args:
                        arg_strategy = dp_strategy_map[arg]
                        assert isinstance(arg_strategy, DataParallelStrategy)
                        input_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)
                        input_specs.append(input_spec)
                    act_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
                    op_strategy = PlacementStrategy(output_spec=act_spec, input_specs=input_specs)
                    dp_strategy_map[node] = DataParallelStrategy(NodeType.ACT, [op_strategy])
        elif node.op == 'output':
            dp_strategy_map[node] = DataParallelStrategy(NodeType.NON_TENSOR, [])
        else:
            raise RuntimeError(f'op code {node.op} not supported')
    return dp_strategy_map