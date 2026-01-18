import warnings
from functools import partial
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNet_Weights, GoogLeNetOutputs, Inception, InceptionAux
from .utils import _fuse_modules, _replace_relu, quantize_model
@register_model(name='quantized_googlenet')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1 if kwargs.get('quantize', False) else GoogLeNet_Weights.IMAGENET1K_V1))
def googlenet(*, weights: Optional[Union[GoogLeNet_QuantizedWeights, GoogLeNet_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableGoogLeNet:
    """GoogLeNet (Inception v1) model architecture from `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`__.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.GoogLeNet_QuantizedWeights` or :class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.GoogLeNet_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableGoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/googlenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.GoogLeNet_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
        :noindex:
    """
    weights = (GoogLeNet_QuantizedWeights if quantize else GoogLeNet_Weights).verify(weights)
    original_aux_logits = kwargs.get('aux_logits', False)
    if weights is not None:
        if 'transform_input' not in kwargs:
            _ovewrite_named_param(kwargs, 'transform_input', True)
        _ovewrite_named_param(kwargs, 'aux_logits', True)
        _ovewrite_named_param(kwargs, 'init_weights', False)
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'fbgemm')
    model = QuantizableGoogLeNet(**kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        else:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them')
    return model