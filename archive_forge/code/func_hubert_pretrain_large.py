import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def hubert_pretrain_large(encoder_projection_dropout: float=0.0, encoder_attention_dropout: float=0.0, encoder_ff_interm_dropout: float=0.0, encoder_dropout: float=0.0, encoder_layer_drop: float=0.0, mask_prob: float=0.8, mask_channel_prob: float=0.0, mask_channel_length: int=10, feature_grad_mult: Optional[float]=None) -> HuBERTPretrainModel:
    """Builds "large" :class:`HuBERTPretrainModel` from *HuBERT* :cite:`hsu2021hubert` for pretraining.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_attention_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_layer_drop (float):
            See :py:func:`hubert_pretrain_model`.
        mask_prob (float):
            See :py:func:`hubert_pretrain_model`.
        mask_channel_prob (float):
            See :py:func:`hubert_pretrain_model`.
        mask_channel_length (int):
            See :py:func:`hubert_pretrain_model`.
        feature_grad_mult (float or None):
            See :py:func:`hubert_pretrain_model`.

    Returns:
        HuBERTPretrainModel:
            The resulting model.
    """
    return hubert_pretrain_model(extractor_mode='layer_norm', extractor_conv_layer_config=None, extractor_conv_bias=False, encoder_embed_dim=1024, encoder_projection_dropout=encoder_projection_dropout, encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16, encoder_num_layers=24, encoder_num_heads=16, encoder_attention_dropout=encoder_attention_dropout, encoder_ff_interm_features=4096, encoder_ff_interm_dropout=encoder_ff_interm_dropout, encoder_dropout=encoder_dropout, encoder_layer_norm_first=True, encoder_layer_drop=encoder_layer_drop, mask_prob=mask_prob, mask_selection='static', mask_other=0.0, mask_length=10, no_mask_overlap=False, mask_min_space=1, mask_channel_prob=mask_channel_prob, mask_channel_selection='static', mask_channel_other=0.0, mask_channel_length=mask_channel_length, no_mask_channel_overlap=False, mask_channel_min_space=1, skip_masked=False, skip_nomask=False, num_classes=500, final_dim=768, feature_grad_mult=feature_grad_mult)