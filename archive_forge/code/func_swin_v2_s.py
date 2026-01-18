import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', Swin_V2_S_Weights.IMAGENET1K_V1))
def swin_v2_s(*, weights: Optional[Swin_V2_S_Weights]=None, progress: bool=True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_S_Weights
        :members:
    """
    weights = Swin_V2_S_Weights.verify(weights)
    return _swin_transformer(patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[8, 8], stochastic_depth_prob=0.3, weights=weights, progress=progress, block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2, **kwargs)