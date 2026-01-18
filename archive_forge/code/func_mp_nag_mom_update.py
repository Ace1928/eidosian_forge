from ._internal import NDArrayBase
from ..base import _Null
def mp_nag_mom_update(weight=None, grad=None, mom=None, weight32=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Update function for multi-precision Nesterov Accelerated Gradient( NAG) optimizer.


    Defined in ../src/operator/optimizer_op.cc:L744

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    mom : NDArray
        Momentum
    weight32 : NDArray
        Weight32
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

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)