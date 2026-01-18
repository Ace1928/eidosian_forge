import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def squim_objective_model(feat_dim: int, win_len: int, d_model: int, nhead: int, hidden_dim: int, num_blocks: int, rnn_type: str, chunk_size: int, chunk_stride: Optional[int]=None) -> SquimObjective:
    """Build a custome :class:`torchaudio.prototype.models.SquimObjective` model.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module.
        win_len (int): Kernel size in the Encoder module.
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        hidden_dim (int): Hidden dimension in the RNN layer of DPRNN.
        num_blocks (int): Number of DPRNN layers.
        rnn_type (str): Type of RNN in DPRNN. Valid options are ["RNN", "LSTM", "GRU"].
        chunk_size (int): Chunk size of input for DPRNN.
        chunk_stride (int or None, optional): Stride of chunk input for DPRNN.
    """
    if chunk_stride is None:
        chunk_stride = chunk_size // 2
    encoder = Encoder(feat_dim, win_len)
    dprnn = DPRNN(feat_dim, hidden_dim, num_blocks, rnn_type, d_model, chunk_size, chunk_stride)
    branches = nn.ModuleList([_create_branch(d_model, nhead, 'stoi'), _create_branch(d_model, nhead, 'pesq'), _create_branch(d_model, nhead, 'sisdr')])
    return SquimObjective(encoder, dprnn, branches)