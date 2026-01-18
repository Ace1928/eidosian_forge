from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor, _validate_trainable_layers
from .generalized_rcnn import GeneralizedRCNN
from .roi_heads import RoIHeads
from .rpn import RegionProposalNetwork, RPNHead
from .transform import GeneralizedRCNNTransform
@register_model()
@handle_legacy_interface(weights=('pretrained', FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1), weights_backbone=('pretrained_backbone', MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def fasterrcnn_mobilenet_v3_large_320_fpn(*, weights: Optional[FasterRCNN_MobileNet_V3_Large_320_FPN_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[MobileNet_V3_Large_Weights]=MobileNet_V3_Large_Weights.IMAGENET1K_V1, trainable_backbone_layers: Optional[int]=None, **kwargs: Any) -> FasterRCNN:
    """
    Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tuned for mobile use cases.

    .. betastatus:: detection module

    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 6, with 6 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        :members:
    """
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
    defaults = {'min_size': 320, 'max_size': 640, 'rpn_pre_nms_top_n_test': 150, 'rpn_post_nms_top_n_test': 150, 'rpn_score_thresh': 0.05}
    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(weights=weights, progress=progress, num_classes=num_classes, weights_backbone=weights_backbone, trainable_backbone_layers=trainable_backbone_layers, **kwargs)