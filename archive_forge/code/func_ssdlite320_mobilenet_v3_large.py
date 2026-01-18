import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .. import mobilenet
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .ssd import SSD, SSDScoringHead
@register_model()
@handle_legacy_interface(weights=('pretrained', SSDLite320_MobileNet_V3_Large_Weights.COCO_V1), weights_backbone=('pretrained_backbone', MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def ssdlite320_mobilenet_v3_large(*, weights: Optional[SSDLite320_MobileNet_V3_Large_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[MobileNet_V3_Large_Weights]=MobileNet_V3_Large_Weights.IMAGENET1K_V1, trainable_backbone_layers: Optional[int]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, **kwargs: Any) -> SSD:
    """SSDlite model architecture with input size 320x320 and a MobileNetV3 Large backbone, as
    described at `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ and
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`__.

    .. betastatus:: detection module

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model
            (including the background).
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers
            starting from final block. Valid values are between 0 and 6, with 6 meaning all
            backbone layers are trainable. If ``None`` is passed (the default) this value is
            set to 6.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
        **kwargs: parameters passed to the ``torchvision.models.detection.ssd.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights
        :members:
    """
    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
    if 'size' in kwargs:
        warnings.warn('The size of the model is already fixed; ignoring the parameter.')
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
    elif num_classes is None:
        num_classes = 91
    trainable_backbone_layers = _validate_trainable_layers(weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6)
    reduce_tail = weights_backbone is None
    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = mobilenet_v3_large(weights=weights_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs)
    if weights_backbone is None:
        _normal_init(backbone)
    backbone = _mobilenet_extractor(backbone, trainable_backbone_layers, norm_layer)
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError(f'The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}')
    defaults = {'score_thresh': 0.001, 'nms_thresh': 0.55, 'detections_per_img': 300, 'topk_candidates': 300, 'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.5, 0.5, 0.5]}
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, size, num_classes, head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model