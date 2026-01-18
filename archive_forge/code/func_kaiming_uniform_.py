import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import (
@_sharded_op_impl(torch.nn.init.kaiming_uniform_)
def kaiming_uniform_(types, args=(), kwargs=None, pg=None):
    """
    Fills the Tensors in tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-\\text{bound}, \\text{bound})` where
    .. math::
        \\text{bound} = \\text{gain} \\times \\sqrt{\\frac{3}{\\text{fan\\_mode}}}
    Also known as He initialization.
    Args:
        tensor: tensor sharded across devices
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    validate_param(kwargs, 'kwargs')
    sharded_tensor = kwargs['tensor']
    validate_param(sharded_tensor, 'tensor')
    a = kwargs['a']
    validate_param(a, 'a')
    mode = kwargs['mode']
    validate_param(mode, 'mode')
    nonlinearity = kwargs['nonlinearity']
    validate_param(nonlinearity, 'nonlinearity')
    for shard in sharded_tensor.local_shards():
        torch.nn.init.kaiming_uniform_(shard.tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    return sharded_tensor