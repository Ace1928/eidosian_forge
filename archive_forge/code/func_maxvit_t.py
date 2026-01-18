import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
@register_model()
@handle_legacy_interface(weights=('pretrained', MaxVit_T_Weights.IMAGENET1K_V1))
def maxvit_t(*, weights: Optional[MaxVit_T_Weights]=None, progress: bool=True, **kwargs: Any) -> MaxVit:
    """
    Constructs a maxvit_t architecture from
    `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_.

    Args:
        weights (:class:`~torchvision.models.MaxVit_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MaxVit_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.maxvit.MaxVit``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MaxVit_T_Weights
        :members:
    """
    weights = MaxVit_T_Weights.verify(weights)
    return _maxvit(stem_channels=64, block_channels=[64, 128, 256, 512], block_layers=[2, 2, 5, 2], head_dim=32, stochastic_depth_prob=0.2, partition_size=7, weights=weights, progress=progress, **kwargs)