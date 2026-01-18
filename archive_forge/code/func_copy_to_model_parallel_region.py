from typing import Optional, Tuple
import torch
import torch.distributed
def copy_to_model_parallel_region(x: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(x, process_group)