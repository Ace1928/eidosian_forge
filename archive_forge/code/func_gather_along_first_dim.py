from typing import Optional, Tuple
import torch
import torch.distributed
def gather_along_first_dim(input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    output, handle = gather_along_first_dim_async(input_, process_group=process_group)
    if handle is not None:
        handle.wait()
    return output