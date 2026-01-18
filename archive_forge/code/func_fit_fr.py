from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def fit_fr(self, data, *args, **kwds):
    """estimate distribution parameters by MLE taking some parameters as fixed

    Parameters
    ----------
    data : ndarray, 1d
        data for which the distribution parameters are estimated,
    args : list ? check
        starting values for optimization
    kwds :

      - 'frozen' : array_like
           values for frozen distribution parameters and, for elements with
           np.nan, the corresponding parameter will be estimated

    Returns
    -------
    argest : ndarray
        estimated parameters


    Examples
    --------
    generate random sample
    >>> np.random.seed(12345)
    >>> x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)

    estimate all parameters
    >>> stats.gamma.fit(x)
    array([ 2.0243194 ,  0.20395655,  1.44411371])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
    array([ 2.0243194 ,  0.20395655,  1.44411371])

    keep loc fixed, estimate shape and scale parameters
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, np.nan])
    array([ 2.45603985,  1.27333105])

    keep loc and scale fixed, estimate shape parameter
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    array([ 3.00048828])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2])
    array([ 2.57792969])

    estimate only scale parameter for fixed shape and loc
    >>> stats.gamma.fit_fr(x, frozen=[2.5, 0.0, np.nan])
    array([ 1.25087891])

    Notes
    -----
    self is an instance of a distribution class. This can be attached to
    scipy.stats.distributions.rv_continuous

    *Todo*

    * check if docstring is correct
    * more input checking, args is list ? might also apply to current fit method

    """
    loc0, scale0 = lmap(kwds.get, ['loc', 'scale'], [0.0, 1.0])
    Narg = len(args)
    if Narg == 0 and hasattr(self, '_fitstart'):
        x0 = self._fitstart(data)
    elif Narg > self.numargs:
        raise ValueError('Too many input arguments.')
    else:
        args += (1.0,) * (self.numargs - Narg)
        x0 = args + (loc0, scale0)
    if 'frozen' in kwds:
        frmask = np.array(kwds['frozen'])
        if len(frmask) != self.numargs + 2:
            raise ValueError('Incorrect number of frozen arguments.')
        else:
            for n in range(len(frmask)):
                if isinstance(frmask[n], np.ndarray) and frmask[n].size == 1:
                    frmask[n] = frmask[n].item()
            frmask = frmask.astype(np.float64)
            x0 = np.array(x0)[np.isnan(frmask)]
    else:
        frmask = None
    return optimize.fmin(self.nnlf_fr, x0, args=(np.ravel(data), frmask), disp=0)