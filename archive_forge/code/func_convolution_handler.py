from typing import cast, Dict, List, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
def convolution_handler(op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> object:
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, 'output sharding should not be None'
    local_results = tp_convolution(op_call, tuple(op_info.local_args), op_info.local_kwargs)
    return dtensor.DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)