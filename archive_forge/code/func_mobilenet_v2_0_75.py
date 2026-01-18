import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
def mobilenet_v2_0_75(**kwargs):
    """MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet_v2(0.75, **kwargs)