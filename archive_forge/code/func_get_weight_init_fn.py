import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def get_weight_init_fn(init_choice: xFormerWeightInit):
    """
    Provide the xFormers factory with weight init routines.

    Supported initializations are:
    - Small: follow the method outlined in `Transformer Without Tears`_
    - ViT: follow the initialization in the reference ViT_ codebase
    - Timm: follow the initialization in the reference Timm_ codebase
    - Moco: follow the initialization in the reference MocoV3_ codebase

    .. _ViT: https://github.com/google-research/vision_transformer
    .. _Timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    .. _MocoV3: https://github.com/facebookresearch/moco-v3
    """
    return {xFormerWeightInit.Timm: _init_weights_vit_timm, xFormerWeightInit.ViT: _init_weights_vit_jax, xFormerWeightInit.Moco: _init_weights_vit_moco, xFormerWeightInit.Small: _init_weights_small}[init_choice]