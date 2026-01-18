import warnings
from typing import Callable, Dict, List, Optional, Union
from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from .. import mobilenet, resnet
from .._api import _get_enum_from_fn, WeightsEnum
from .._utils import handle_legacy_interface, IntermediateLayerGetter
@handle_legacy_interface(weights=('pretrained', lambda kwargs: _get_enum_from_fn(mobilenet.__dict__[kwargs['backbone_name']])['IMAGENET1K_V1']))
def mobilenet_backbone(*, backbone_name: str, weights: Optional[WeightsEnum], fpn: bool, norm_layer: Callable[..., nn.Module]=misc_nn_ops.FrozenBatchNorm2d, trainable_layers: int=2, returned_layers: Optional[List[int]]=None, extra_blocks: Optional[ExtraFPNBlock]=None) -> nn.Module:
    backbone = mobilenet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _mobilenet_extractor(backbone, fpn, trainable_layers, returned_layers, extra_blocks)