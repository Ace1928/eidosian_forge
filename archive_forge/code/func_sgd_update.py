from ._internal import NDArrayBase
from ..base import _Null
def sgd_update(weight=None, grad=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, out=None, name=None, **kwargs):
    """Update function for Stochastic Gradient Descent (SGD) optimizer.

    It updates the weights using::

     weight = weight - learning_rate * (gradient + wd * weight)

    However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,
    only the row slices whose indices appear in grad.indices are updated::

     for row in gradient.indices:
         weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])



    Defined in ../src/operator/optimizer_op.cc:L523

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    lr : float, required
        Learning rate
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)