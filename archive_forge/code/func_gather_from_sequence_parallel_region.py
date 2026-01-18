from typing import Optional, Tuple
import torch
import torch.distributed
def gather_from_sequence_parallel_region(x: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(x, process_group)