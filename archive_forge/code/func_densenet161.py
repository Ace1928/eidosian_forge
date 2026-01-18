import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet161_Weights.IMAGENET1K_V1))
def densenet161(*, weights: Optional[DenseNet161_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    """Densenet-161 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet161_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet161_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet161_Weights
        :members:
    """
    weights = DenseNet161_Weights.verify(weights)
    return _densenet(48, (6, 12, 36, 24), 96, weights, progress, **kwargs)