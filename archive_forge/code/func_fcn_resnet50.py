from functools import partial
from typing import Any, Optional
from torch import nn
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
@register_model()
@handle_legacy_interface(weights=('pretrained', FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=('pretrained_backbone', ResNet50_Weights.IMAGENET1K_V1))
def fcn_resnet50(*, weights: Optional[FCN_ResNet50_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, aux_loss: Optional[bool]=None, weights_backbone: Optional[ResNet50_Weights]=ResNet50_Weights.IMAGENET1K_V1, **kwargs: Any) -> FCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
        :members:
    """
    weights = FCN_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
        aux_loss = _ovewrite_value_param('aux_loss', aux_loss, True)
    elif num_classes is None:
        num_classes = 21
    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model