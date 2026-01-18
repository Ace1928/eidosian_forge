import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def compute_correct_candidates(candidate_starts, candidate_ends, gold_starts, gold_ends):
    """Compute correct span."""
    is_gold_start = torch.eq(torch.unsqueeze(torch.unsqueeze(candidate_starts, 0), 0), torch.unsqueeze(gold_starts, -1))
    is_gold_end = torch.eq(torch.unsqueeze(torch.unsqueeze(candidate_ends, 0), 0), torch.unsqueeze(gold_ends, -1))
    return torch.any(torch.logical_and(is_gold_start, is_gold_end), 1)