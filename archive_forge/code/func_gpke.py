import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def gpke(bw, data, data_predict, var_type, ckertype='gaussian', okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    """
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw : 1-D ndarray
        The user-specified bandwidth parameters.
    data : 1D or 2-D ndarray
        The training data.
    data_predict : 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type : str, optional
        The variable type (continuous, ordered, unordered).
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    tosum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.

    Returns
    -------
    dens : array_like
        The generalized product kernel density estimator.

    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:

    .. math:: f(x)=\\frac{1}{nh_{1}...h_{q}}\\sum_{i=1}^
                        {n}K\\left(\\frac{X_{i}-x}{h}\\right)

    where

    .. math:: K\\left(\\frac{X_{i}-x}{h}\\right) =
                k\\left( \\frac{X_{i1}-x_{1}}{h_{1}}\\right)\\times
                k\\left( \\frac{X_{i2}-x_{2}}{h_{2}}\\right)\\times...\\times
                k\\left(\\frac{X_{iq}-x_{q}}{h_{q}}\\right)
    """
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)
    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bw[ii], data[:, ii], data_predict[ii])
    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bw[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens