import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def squim_objective_base() -> SquimObjective:
    """Build :class:`torchaudio.prototype.models.SquimObjective` model with default arguments."""
    return squim_objective_model(feat_dim=256, win_len=64, d_model=256, nhead=4, hidden_dim=256, num_blocks=2, rnn_type='LSTM', chunk_size=71)