from typing import Optional, Tuple
import torch
import torch.distributed
def scatter_to_sequence_parallel_region(x: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(x, process_group)