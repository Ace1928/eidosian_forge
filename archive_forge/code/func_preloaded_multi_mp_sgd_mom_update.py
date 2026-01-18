from ._internal import NDArrayBase
from ..base import _Null
def preloaded_multi_mp_sgd_mom_update(*data, **kwargs):
    """Momentum update function for multi-precision Stochastic Gradient Descent (SGD) optimizer.

    Momentum update has better convergence rates on neural networks. Mathematically it looks
    like below:

    .. math::

      v_1 = \\alpha * \\nabla J(W_0)\\\\
      v_t = \\gamma v_{t-1} - \\alpha * \\nabla J(W_{t-1})\\\\
      W_t = W_{t-1} + v_t

    It updates the weights using::

      v = momentum * v - learning_rate * gradient
      weight += v

    Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.



    Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L199

    Parameters
    ----------
    data : NDArray[]
        Weights, gradients, momentums, learning rates and weight decays
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
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