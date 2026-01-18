from functools import partial
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead
@register_model()
@handle_legacy_interface(weights=('pretrained', DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=('pretrained_backbone', ResNet101_Weights.IMAGENET1K_V1))
def deeplabv3_resnet101(*, weights: Optional[DeepLabV3_ResNet101_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, aux_loss: Optional[bool]=None, weights_backbone: Optional[ResNet101_Weights]=ResNet101_Weights.IMAGENET1K_V1, **kwargs: Any) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
        aux_loss = _ovewrite_value_param('aux_loss', aux_loss, True)
    elif num_classes is None:
        num_classes = 21
    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model