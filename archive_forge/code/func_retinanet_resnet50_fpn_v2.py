import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ...ops import boxes as box_ops, misc as misc_nn_ops, sigmoid_focal_loss
from ...ops.feature_pyramid_network import LastLevelP6P7
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from . import _utils as det_utils
from ._utils import _box_loss, overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
@register_model()
@handle_legacy_interface(weights=('pretrained', RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1), weights_backbone=('pretrained_backbone', ResNet50_Weights.IMAGENET1K_V1))
def retinanet_resnet50_fpn_v2(*, weights: Optional[RetinaNet_ResNet50_FPN_V2_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[ResNet50_Weights]=None, trainable_backbone_layers: Optional[int]=None, **kwargs: Any) -> RetinaNet:
    """
    Constructs an improved RetinaNet model with a ResNet-50-FPN backbone.

    .. betastatus:: detection module

    Reference: `Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection
    <https://arxiv.org/abs/1912.02424>`_.

    :func:`~torchvision.models.detection.retinanet_resnet50_fpn` for more details.

    Args:
        weights (:class:`~torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for
            the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.RetinaNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights
        :members:
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
    elif num_classes is None:
        num_classes = 91
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256))
    anchor_generator = _default_anchorgen()
    head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes, norm_layer=partial(nn.GroupNorm, 32))
    head.regression_head._loss_type = 'giou'
    model = RetinaNet(backbone, num_classes, anchor_generator=anchor_generator, head=head, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model