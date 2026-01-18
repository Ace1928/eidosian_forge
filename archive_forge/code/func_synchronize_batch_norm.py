import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
def synchronize_batch_norm(self):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_rank() > -1)):
        raise RuntimeError('Make sure torch.distributed is set up properly.')
    self_rank = torch.distributed.get_rank()
    node_size = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    if not world_size % node_size == 0:
        raise RuntimeError('Make sure the number of processes can be divided by the number of nodes')
    node_global_ranks = [list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)]
    sync_bn_groups = [torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)]
    node_rank = self_rank // node_size
    self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])