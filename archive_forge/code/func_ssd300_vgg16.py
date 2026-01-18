import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...ops import boxes as box_ops
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..vgg import VGG, vgg16, VGG16_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
@register_model()
@handle_legacy_interface(weights=('pretrained', SSD300_VGG16_Weights.COCO_V1), weights_backbone=('pretrained_backbone', VGG16_Weights.IMAGENET1K_FEATURES))
def ssd300_vgg16(*, weights: Optional[SSD300_VGG16_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[VGG16_Weights]=VGG16_Weights.IMAGENET1K_FEATURES, trainable_backbone_layers: Optional[int]=None, **kwargs: Any) -> SSD:
    """The SSD300 model is based on the `SSD: Single Shot MultiBox Detector
    <https://arxiv.org/abs/1512.02325>`_ paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes, but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        >>> model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSD300_VGG16_Weights`, optional): The pretrained
                weights to use. See
                :class:`~torchvision.models.detection.SSD300_VGG16_Weights`
                below for more details, and possible values. By default, no
                pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr
            Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.VGG16_Weights`, optional): The pretrained weights for the
            backbone
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 4.
        **kwargs: parameters passed to the ``torchvision.models.detection.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSD300_VGG16_Weights
        :members:
    """
    weights = SSD300_VGG16_Weights.verify(weights)
    weights_backbone = VGG16_Weights.verify(weights_backbone)
    if 'size' in kwargs:
        warnings.warn('The size of the model is already fixed; ignoring the parameter.')
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
    elif num_classes is None:
        num_classes = 91
    trainable_backbone_layers = _validate_trainable_layers(weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 4)
    backbone = vgg16(weights=weights_backbone, progress=progress)
    backbone = _vgg_extractor(backbone, False, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]], scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], steps=[8, 16, 32, 64, 100, 300])
    defaults = {'image_mean': [0.48235, 0.45882, 0.40784], 'image_std': [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]}
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model