from ._internal import NDArrayBase
from ..base import _Null
def ftrl_update(weight=None, grad=None, z=None, n=None, lr=_Null, lamda1=_Null, beta=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Update function for Ftrl optimizer.
    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    It updates the weights using::

     rescaled_grad = clip(grad * rescale_grad, clip_gradient)
     z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
     n += rescaled_grad**2
     w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

    If w, z and n are all of ``row_sparse`` storage type,
    only the row slices whose indices appear in grad.indices are updated (for w, z and n)::

     for row in grad.indices:
         rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
         z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
         n[row] += rescaled_grad[row]**2
         w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)



    Defined in ../src/operator/optimizer_op.cc:L875

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    z : NDArray
        z
    n : NDArray
        Square of grad
    lr : float, required
        Learning rate
    lamda1 : float, optional, default=0.00999999978
        The L1 regularization coefficient.
    beta : float, optional, default=1
        Per-Coordinate Learning Rate beta.
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