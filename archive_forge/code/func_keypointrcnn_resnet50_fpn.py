from typing import Any, Optional
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_PERSON_CATEGORIES, _COCO_PERSON_KEYPOINT_NAMES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .faster_rcnn import FasterRCNN
@register_model()
@handle_legacy_interface(weights=('pretrained', lambda kwargs: KeypointRCNN_ResNet50_FPN_Weights.COCO_LEGACY if kwargs['pretrained'] == 'legacy' else KeypointRCNN_ResNet50_FPN_Weights.COCO_V1), weights_backbone=('pretrained_backbone', ResNet50_Weights.IMAGENET1K_V1))
def keypointrcnn_resnet50_fpn(*, weights: Optional[KeypointRCNN_ResNet50_FPN_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, num_keypoints: Optional[int]=None, weights_backbone: Optional[ResNet50_Weights]=ResNet50_Weights.IMAGENET1K_V1, trainable_backbone_layers: Optional[int]=None, **kwargs: Any) -> KeypointRCNN:
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    .. betastatus:: detection module

    Reference: `Mask R-CNN <https://arxiv.org/abs/1703.06870>`__.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        num_keypoints (int, optional): number of keypoints
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.

    .. autoclass:: torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights
        :members:
    """
    weights = KeypointRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
        num_keypoints = _ovewrite_value_param('num_keypoints', num_keypoints, len(weights.meta['keypoint_names']))
    else:
        if num_classes is None:
            num_classes = 2
        if num_keypoints is None:
            num_keypoints = 17
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d
    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = KeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == KeypointRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)
    return model