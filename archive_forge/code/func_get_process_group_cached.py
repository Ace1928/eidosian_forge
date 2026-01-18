from enum import Enum
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence
import torch
import torch.distributed as dist
import torch.nn.functional as F
def get_process_group_cached(name: ProcessGroupName=ProcessGroupName.default, ranks: Optional[Sequence[int]]=None) -> 'ProcessGroup':
    """
    Singleton PyTorch distributed group cache. Inspired by the code from fairseq.

    Just like torch.distributed.new_group, this method needs to be called on all ranks
    at the same time when a new group is created. This is true for all ranks irrespective
    of their group membership status.

    For FSDP, it is important to use the same group between outer and inner FSDP instances,
    otherwise, inner FSDP instances will not share the gradient reduction bucket buffer with
    the root instance. This will result in increased GPU memory utilization.

    Each separate process group also uses separate NCCL library instances, which will have
    a significant effect on GPU memory use if too many process groups are created and used.
    Setting NCCL_BUFFSIZE=102400 env variable is a useful technique to check if the NCCL
    memory is causing GPU OOM. Note, the NCCL buffers are not allocated
    through the PyTorch caching allocator, therefore, you may see GPU OOM even when
    torch.cuda.reserved_memory() is still way below the total amount of GPU memory.

    Extra process groups can also reduce training speed (observed on VISSL models).

    Args:
        name ProcessGroupName:
            There are two process groups when reduce_scatter overlap is enabled. The "default" process group is the
            default process group. The other group is "reduce_scatter" group.
            Default: ProcessGroupName.default
        ranks (Optional[List[int]]):
            Ranks requested in the target group. None for all ranks.
            Default: None

    Returns:
        (ProcessGroup):
            Return the requested process group. Throws RuntimeError if torch.distributed module is not yet initialized.
    """
    if not dist.is_initialized():
        if name == ProcessGroupName.reduce_scatter and 'pytest' in sys.modules:
            return None
        else:
            raise RuntimeError('torch.distributed is not yet initialized but process group is requested.')
    if not hasattr(get_process_group_cached, '_global_group_cache'):
        get_process_group_cached._global_group_cache = {}
        cache = get_process_group_cached._global_group_cache
        default_pg = dist.new_group(ranks=ranks)
        cache[None] = default_pg
        cache[ProcessGroupName.default, None] = default_pg
        cache[ProcessGroupName.default, frozenset(list(range(dist.get_world_size())))] = default_pg
    cache = get_process_group_cached._global_group_cache
    if ranks is not None:
        ranks = tuple(sorted(list(set(ranks))))
    if (name, ranks) not in cache:
        cache[name, ranks] = dist.new_group(ranks=ranks)
    return cache[name, ranks]