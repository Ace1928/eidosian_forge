from typing import List, Tuple
import torch
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def foreach_list_strategy(mesh: DeviceMesh, op_schema: OpSchema, linearity: bool=False) -> StrategyType:
    """
    for each list op stratgy mostly follow the same logic as pointwise strategy
    except that it handles list of tensors instead, and normally we don't need to
    handle implicit broadcasting
    """

    def args_tuple_strategies(args_schema: Tuple[object, ...]) -> List[TupleStrategy]:
        first_arg = args_schema[0]
        assert isinstance(first_arg, TupleStrategy)
        strategy_len = len(first_arg.childs)
        tuple_strategies: List[TupleStrategy] = []
        for arg in args_schema:
            if isinstance(arg, TupleStrategy):
                assert len(arg.childs) == strategy_len
                tuple_strategies.append(arg)
            elif isinstance(arg, OpStrategy):
                raise RuntimeError('foreach list op only supports tuple strategy!')
        return tuple_strategies
    args_strategies = args_tuple_strategies(op_schema.args_schema)
    follow_strategy = args_strategies[0]
    foreach_strategy_list = []
    for idx, child_strtgy in enumerate(follow_strategy.childs):
        assert isinstance(child_strtgy, OpStrategy)
        strategies = []
        for strtgy in child_strtgy.strategies:
            spec_to_follow = strtgy.output_spec
            if not linearity:
                assert not is_tensor_partial(spec_to_follow), f'{op_schema.op} does not support operation on partial tensor!'
            redistribute_costs: List[List[float]] = []
            for arg_strtgy in args_strategies:
                child_strtgy = arg_strtgy.childs[idx]
                assert isinstance(child_strtgy, OpStrategy)
                redistribute_costs.append(generate_redistribute_costs(child_strtgy, spec_to_follow))
            strategies.append(PlacementStrategy(output_spec=spec_to_follow, redistribute_cost=redistribute_costs))
        foreach_strategy_list.append(OpStrategy(strategies))
    tup_strategy = TupleStrategy(foreach_strategy_list)
    return tup_strategy