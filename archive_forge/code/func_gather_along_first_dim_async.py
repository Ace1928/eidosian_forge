from typing import Optional, Tuple
import torch
import torch.distributed
def gather_along_first_dim_async(input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    assert input_.is_contiguous()
    mp_size = process_group.size()
    if mp_size == 1:
        return (input_, None)
    output = input_.new_empty((input_.shape[0] * mp_size,) + input_.shape[1:])
    handle = torch.distributed.all_gather_into_tensor(output_tensor=output, input_tensor=input_, group=process_group, async_op=True)
    return (output, handle)