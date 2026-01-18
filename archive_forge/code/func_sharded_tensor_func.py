import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
def sharded_tensor_func(value, pg, device):
    cpu_device = torch.device('cpu')
    output_tensor = _all_gather_sharded_tensor(value, pg, device)
    local_shard_device = value.local_shards()[0].tensor.device if value.local_shards() else cpu_device
    if output_tensor.device != local_shard_device:
        value = output_tensor.to(local_shard_device)
    else:
        value = output_tensor
    return value