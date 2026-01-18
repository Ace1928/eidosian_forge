import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def hubert_base(encoder_projection_dropout: float=0.1, encoder_attention_dropout: float=0.1, encoder_ff_interm_dropout: float=0.0, encoder_dropout: float=0.1, encoder_layer_drop: float=0.05, aux_num_out: Optional[int]=None) -> Wav2Vec2Model:
    """Builds "base" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wav2vec2_model(extractor_mode='group_norm', extractor_conv_layer_config=None, extractor_conv_bias=False, encoder_embed_dim=768, encoder_projection_dropout=encoder_projection_dropout, encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16, encoder_num_layers=12, encoder_num_heads=12, encoder_attention_dropout=encoder_attention_dropout, encoder_ff_interm_features=3072, encoder_ff_interm_dropout=encoder_ff_interm_dropout, encoder_dropout=encoder_dropout, encoder_layer_norm_first=False, encoder_layer_drop=encoder_layer_drop, aux_num_out=aux_num_out)