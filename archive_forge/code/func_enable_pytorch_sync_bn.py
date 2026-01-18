from enum import Enum
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence
import torch
import torch.distributed as dist
import torch.nn.functional as F
def enable_pytorch_sync_bn(module: torch.nn.Module) -> None:
    """Call _specify_ddp_gpu_num for all pytorch SyncBN layers so that it
    is happily running even without DDP. E.g. this is used by FSDP.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm) and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)