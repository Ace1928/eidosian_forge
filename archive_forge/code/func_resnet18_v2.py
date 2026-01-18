import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from .... import base
from .... util import is_np_array
def resnet18_v2(**kwargs):
    """ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 18, **kwargs)