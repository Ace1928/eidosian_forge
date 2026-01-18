from typing import Any, List, Optional
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
def gather_all_tensors(result: Tensor, group: Optional[Any]=None) -> List[Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group where element i corresponds to result tensor from process i

    """
    if group is None:
        group = torch.distributed.group.WORLD
    result = result.contiguous()
    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all((all(ls == max_size) for ls in local_sizes))
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result