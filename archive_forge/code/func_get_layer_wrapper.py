import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers._deprecation_warning import deprecated_function
from xformers.components import (
from xformers.components.attention import AttentionMask
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import build_positional_embedding
from xformers.components.residual import get_deepnorm_coefficients
from xformers.components.simplicial_embedding import SimplicialEmbedding
from xformers.factory.block_configs import (
def get_layer_wrapper(d_model: int, sublayer: nn.Module, residual_norm_style: Optional[ResidualNormStyle], residual: bool, residual_scale: float):
    if residual:
        if residual_norm_style == ResidualNormStyle.Pre:
            return Residual(layer=PreNorm(d_model, sublayer, normalization, use_triton), scale=None)
        elif residual_norm_style == ResidualNormStyle.Post:
            return PostNorm(d_model, Residual(layer=sublayer, scale=None), normalization, use_triton)
        elif residual_norm_style == ResidualNormStyle.DeepNorm:
            return PostNorm(d_model, Residual(layer=sublayer, scale=residual_scale), normalization, use_triton=use_triton)
        else:
            raise ValueError
    return PreNorm(d_model, sublayer, normalization, use_triton) if residual_norm_style == ResidualNormStyle.Pre else PostNorm(d_model, sublayer, normalization, use_triton)