from ._internal import NDArrayBase
from ..base import _Null
def mp_lamb_update_phase1(weight=None, grad=None, mean=None, var=None, weight32=None, beta1=_Null, beta2=_Null, epsilon=_Null, t=_Null, bias_correction=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Mixed Precision version of Phase I of lamb update 
    it performs the following operations and returns g:.

              Link to paper: https://arxiv.org/pdf/1904.00962.pdf

              .. math::
                  \\begin{gather*}
                  grad32 = grad(float16) * rescale_grad
                  if (grad < -clip_gradient)
                  then
                       grad = -clip_gradient
                  if (grad > clip_gradient)
                  then
                       grad = clip_gradient

                  mean = beta1 * mean + (1 - beta1) * grad;
                  variance = beta2 * variance + (1. - beta2) * grad ^ 2;

                  if (bias_correction)
                  then
                       mean_hat = mean / (1. - beta1^t);
                       var_hat = var / (1 - beta2^t);
                       g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight32;
                  else
                       g = mean / (var_data^(1/2) + epsilon) + wd * weight32;
                  \\end{gather*}

          

    Defined in ../src/operator/optimizer_op.cc:L1032

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    mean : NDArray
        Moving mean
    var : NDArray
        Moving variance
    weight32 : NDArray
        Weight32
    beta1 : float, optional, default=0.899999976
        The decay rate for the 1st moment estimates.
    beta2 : float, optional, default=0.999000013
        The decay rate for the 2nd moment estimates.
    epsilon : float, optional, default=9.99999997e-07
        A small constant for numerical stability.
    t : int, required
        Index update count.
    bias_correction : boolean, optional, default=1
        Whether to use bias correction.
    wd : float, required
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