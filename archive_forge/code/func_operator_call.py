import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
@staticmethod
def operator_call(sparse_query, indices, query_num_block, key_num_block):
    batch_size, num_block, block_size, _ = sparse_query.size()
    if len(sparse_query.size()) != 4:
        raise ValueError('sparse_query must be a 4-dimensional tensor.')
    if len(indices.size()) != 2:
        raise ValueError('indices must be a 2-dimensional tensor.')
    _, _, block_size, _ = sparse_query.size()
    batch_size, num_block = indices.size()
    sparse_query = sparse_query.sum(dim=2).reshape(batch_size * num_block, block_size)
    batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
    global_idxes = (torch.div(indices, key_num_block, rounding_mode='floor').long() + batch_idx[:, None] * query_num_block).reshape(batch_size * num_block)
    temp = torch.zeros((batch_size * query_num_block, block_size), dtype=sparse_query.dtype, device=sparse_query.device)
    output = temp.index_add(0, global_idxes, sparse_query).reshape(batch_size, query_num_block, block_size)
    output = output.reshape(batch_size, query_num_block * block_size)
    return output