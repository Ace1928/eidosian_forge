from typing import Optional, Tuple
import torch
import torch.distributed
def reduce_scatter_along_first_dim_async(input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    assert input_.is_contiguous()
    mp_size = process_group.size()
    if mp_size == 1:
        return (input_, None)
    assert input_.shape[0] % mp_size == 0
    output = input_.new_empty((input_.shape[0] // mp_size,) + input_.shape[1:])
    handle = torch.distributed.reduce_scatter_tensor(output=output, input=input_, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True)
    return (output, handle)