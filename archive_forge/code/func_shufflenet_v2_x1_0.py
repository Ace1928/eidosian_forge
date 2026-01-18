from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import shufflenetv2
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..shufflenetv2 import (
from .utils import _fuse_modules, _replace_relu, quantize_model
@register_model(name='quantized_shufflenet_v2_x1_0')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: ShuffleNet_V2_X1_0_QuantizedWeights.IMAGENET1K_FBGEMM_V1 if kwargs.get('quantize', False) else ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1))
def shufflenet_v2_x1_0(*, weights: Optional[Union[ShuffleNet_V2_X1_0_QuantizedWeights, ShuffleNet_V2_X1_0_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/abs/1807.11164>`__.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights` or :class:`~torchvision.models.ShuffleNet_V2_X1_0_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.
            Default is True.
        quantize (bool, optional): If True, return a quantized version of the model.
            Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/shufflenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.ShuffleNet_V2_X1_0_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.ShuffleNet_V2_X1_0_Weights
        :members:
        :noindex:
    """
    weights = (ShuffleNet_V2_X1_0_QuantizedWeights if quantize else ShuffleNet_V2_X1_0_Weights).verify(weights)
    return _shufflenetv2([4, 8, 4], [24, 116, 232, 464, 1024], weights=weights, progress=progress, quantize=quantize, **kwargs)