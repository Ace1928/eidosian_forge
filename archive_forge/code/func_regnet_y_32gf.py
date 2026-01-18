import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', RegNet_Y_32GF_Weights.IMAGENET1K_V1))
def regnet_y_32gf(*, weights: Optional[RegNet_Y_32GF_Weights]=None, progress: bool=True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_32GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_32GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_32GF_Weights
        :members:
    """
    weights = RegNet_Y_32GF_Weights.verify(weights)
    params = BlockParams.from_init_params(depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs)
    return _regnet(params, weights, progress, **kwargs)