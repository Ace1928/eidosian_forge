from ._internal import NDArrayBase
from ..base import _Null
def sgd_mom_update(weight=None, grad=None, mom=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, out=None, name=None, **kwargs):
    """Momentum update function for Stochastic Gradient Descent (SGD) optimizer.

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

    However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage
    type is the same as momentum's storage type,
    only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::

      for row in gradient.indices:
          v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
          weight[row] += v[row]



    Defined in ../src/operator/optimizer_op.cc:L564

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    mom : NDArray
        Momentum
    lr : float, required
        Learning rate
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse and both weight and momentum have the same stype

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)