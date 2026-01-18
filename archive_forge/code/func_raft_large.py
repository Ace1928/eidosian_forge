from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
@register_model()
@handle_legacy_interface(weights=('pretrained', Raft_Large_Weights.C_T_SKHT_V2))
def raft_large(*, weights: Optional[Raft_Large_Weights]=None, progress=True, **kwargs) -> RAFT:
    """RAFT model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`_.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.models.optical_flow.Raft_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.optical_flow.Raft_Large_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.optical_flow.RAFT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.optical_flow.Raft_Large_Weights
        :members:
    """
    weights = Raft_Large_Weights.verify(weights)
    return _raft(weights=weights, progress=progress, feature_encoder_layers=(64, 64, 96, 128, 256), feature_encoder_block=ResidualBlock, feature_encoder_norm_layer=InstanceNorm2d, context_encoder_layers=(64, 64, 96, 128, 256), context_encoder_block=ResidualBlock, context_encoder_norm_layer=BatchNorm2d, corr_block_num_levels=4, corr_block_radius=4, motion_encoder_corr_layers=(256, 192), motion_encoder_flow_layers=(128, 64), motion_encoder_out_channels=128, recurrent_block_hidden_state_size=128, recurrent_block_kernel_size=((1, 5), (5, 1)), recurrent_block_padding=((0, 2), (2, 0)), flow_head_hidden_size=256, use_mask_predictor=True, **kwargs)