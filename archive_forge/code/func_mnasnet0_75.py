import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', MNASNet0_75_Weights.IMAGENET1K_V1))
def mnasnet0_75(*, weights: Optional[MNASNet0_75_Weights]=None, progress: bool=True, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 0.75 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet0_75_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet0_75_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet0_75_Weights
        :members:
    """
    weights = MNASNet0_75_Weights.verify(weights)
    return _mnasnet(0.75, weights, progress, **kwargs)