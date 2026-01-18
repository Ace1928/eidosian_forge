from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
def emformer_rnnt_base(num_symbols: int) -> RNNT:
    """Builds basic version of Emformer-based :class:`~torchaudio.models.RNNT`.

    Args:
        num_symbols (int): The size of target token lexicon.

    Returns:
        RNNT:
            Emformer RNN-T model.
    """
    return emformer_rnnt_model(input_dim=80, encoding_dim=1024, num_symbols=num_symbols, segment_length=16, right_context_length=4, time_reduction_input_dim=128, time_reduction_stride=4, transformer_num_heads=8, transformer_ffn_dim=2048, transformer_num_layers=20, transformer_dropout=0.1, transformer_activation='gelu', transformer_left_context_length=30, transformer_max_memory_size=0, transformer_weight_init_scale_strategy='depthwise', transformer_tanh_on_mem=True, symbol_embedding_dim=512, num_lstm_layers=3, lstm_layer_norm=True, lstm_layer_norm_epsilon=0.001, lstm_dropout=0.3)