from ._internal import NDArrayBase
from ..base import _Null
def preloaded_multi_sgd_update(*data, **kwargs):
    """Update function for Stochastic Gradient Descent (SDG) optimizer.

    It updates the weights using::

     weight = weight - learning_rate * (gradient + wd * weight)



    Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L41

    Parameters
    ----------
    data : NDArray[]
        Weights, gradients, learning rates and weight decays
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    num_weights : int, optional, default='1'
        Number of updated weights.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)