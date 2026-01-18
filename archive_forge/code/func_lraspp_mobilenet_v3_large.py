from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
@register_model()
@handle_legacy_interface(weights=('pretrained', LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=('pretrained_backbone', MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def lraspp_mobilenet_v3_large(*, weights: Optional[LRASPP_MobileNet_V3_Large_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[MobileNet_V3_Large_Weights]=MobileNet_V3_Large_Weights.IMAGENET1K_V1, **kwargs: Any) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights
        :members:
    """
    if kwargs.pop('aux_loss', False):
        raise NotImplementedError('This model does not use auxiliary loss')
    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
    elif num_classes is None:
        num_classes = 21
    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model