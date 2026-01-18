from functools import partial
from typing import Any, List, Optional, Union
import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from ...ops.misc import Conv2dNormActivation, SqueezeExcitation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..mobilenetv3 import (
from .utils import _fuse_modules, _replace_relu
@register_model(name='quantized_mobilenet_v3_large')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1 if kwargs.get('quantize', False) else MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def mobilenet_v3_large(*, weights: Optional[Union[MobileNet_V3_Large_QuantizedWeights, MobileNet_V3_Large_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableMobileNetV3:
    """
    MobileNetV3 (Large) model from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V3_Large_Weights
        :members:
        :noindex:
    """
    weights = (MobileNet_V3_Large_QuantizedWeights if quantize else MobileNet_V3_Large_Weights).verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('mobilenet_v3_large', **kwargs)
    return _mobilenet_v3_model(inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs)