from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def conformer_wav2vec2_pretrain_base(extractor_input_dim: int=64, extractor_output_dim: int=256, encoder_projection_dropout: float=0.0, mask_prob: float=0.3, mask_length: int=3, num_negatives: int=100, cross_sample_negatives: int=0) -> ConformerWav2Vec2PretrainModel:
    """Build Conformer Wav2Vec2 Model for pre-training with "small" architecture from
    *Conformer-Based Self-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional): Input dimension of the features. (Default: 64)
        extractor_output_dim (int, optional): Output dimension after feature extraction. (Default: 256)
        encoder_projection_dropout (float, optional):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``. (Default: 0.0)
        mask_prob (float, optional):
            Probability for each token to be chosen as start of the span to be masked. (Default: 0.3)
        mask_length (int, optional):
            The lengths of the mask. (Default: 3)
        num_negatives (int, optional):
            Number of sampled negatives. (Default: 0)
        cross_sample_negatives (int, optional):
            Number of cross sampled negatives. (Default: 0)

    Returns:
        ConformerWav2Vec2PretrainModel:
            The resulting model.
    """
    return conformer_wav2vec2_pretrain_model(extractor_input_dim=extractor_input_dim, extractor_output_dim=extractor_output_dim, extractor_stride=4, encoder_embed_dim=256, encoder_projection_dropout=encoder_projection_dropout, encoder_num_layers=12, encoder_num_heads=8, encoder_ff_interm_features=1024, encoder_depthwise_conv_kernel_size=[31] + [15] * 11, encoder_dropout=0.1, encoder_convolution_first=True, encoder_use_group_norm=True, mask_prob=mask_prob, mask_selection='static', mask_other=0.0, mask_length=mask_length, no_mask_overlap=False, mask_min_space=0, mask_channel_prob=0, mask_channel_selection='static', mask_channel_other=0, mask_channel_length=10, no_mask_channel_overlap=False, mask_channel_min_space=1, num_negatives=num_negatives, cross_sample_negatives=cross_sample_negatives)