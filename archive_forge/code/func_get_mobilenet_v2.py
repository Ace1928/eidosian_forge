import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
def get_mobilenet_v2(multiplier, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'), **kwargs):
    """MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = MobileNetV2(multiplier, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_parameters(get_model_file('mobilenetv2_%s' % version_suffix, root=root), ctx=ctx)
    return net