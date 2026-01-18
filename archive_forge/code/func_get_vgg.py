import os
from ....context import cpu
from ....initializer import Xavier
from ...block import HybridBlock
from ... import nn
from .... import base
def get_vgg(num_layers, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'), **kwargs):
    """VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.load_parameters(get_model_file('vgg%d%s' % (num_layers, batch_norm_suffix), root=root), ctx=ctx)
    return net