import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools
def fuse_conv_bn(is_qat, conv, bn):
    """Return the fused the conv and bn modules.
    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert conv.training == bn.training, 'Conv and BN both must be in the same mode (train or eval).'
    fused_module_class_map = {nn.Conv1d: nni.ConvBn1d, nn.Conv2d: nni.ConvBn2d, nn.Conv3d: nni.ConvBn3d}
    if is_qat:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get(type(conv), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn)}')
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)