from ._internal import NDArrayBase
from ..base import _Null
def mp_sgd_update(weight=None, grad=None, weight32=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, out=None, name=None, **kwargs):
    """Updater function for multi-precision sgd optimizer

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        gradient
    weight32 : NDArray
        Weight32
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