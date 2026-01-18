from ._internal import NDArrayBase
from ..base import _Null
def group_adagrad_update(weight=None, grad=None, history=None, lr=_Null, rescale_grad=_Null, clip_gradient=_Null, epsilon=_Null, out=None, name=None, **kwargs):
    """Update function for Group AdaGrad optimizer.

    Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
    and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
    uses only a single learning rate for every row of the parameter array.

    Updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += mean(square(grad), axis=1, keepdims=True)
        div = grad / sqrt(history + float_stable_eps)
        weight -= div * lr

    Weights are updated lazily if the gradient is sparse.

    Note that non-zero values for the weight decay option are not supported.



    Defined in ../src/operator/contrib/optimizer_op.cc:L70

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    history : NDArray
        History
    lr : float, required
        Learning rate
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    epsilon : float, optional, default=9.99999975e-06
        Epsilon for numerical stability

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)