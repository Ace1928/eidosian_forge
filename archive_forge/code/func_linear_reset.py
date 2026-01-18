from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
@deprecate_kwarg('result', 'res')
def linear_reset(res, power=3, test_type='fitted', use_f=False, cov_type='nonrobust', cov_kwargs=None):
    """
    Ramsey's RESET test for neglected nonlinearity

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    power : {int, List[int]}, default 3
        The maximum power to include in the model, if an integer. Includes
        powers 2, 3, ..., power. If an list of integers, includes all powers
        in the list.
    test_type : str, default "fitted"
        The type of augmentation to use:

        * "fitted" : (default) Augment regressors with powers of fitted values.
        * "exog" : Augment exog with powers of exog. Excludes binary
          regressors.
        * "princomp": Augment exog with powers of first principal component of
          exog.
    use_f : bool, default False
        Flag indicating whether an F-test should be used (True) or a
        chi-square test (False).
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    ContrastResults
        Test results for Ramsey's Reset test. See notes for implementation
        details.

    Notes
    -----
    The RESET test uses an augmented regression of the form

    .. math::

       Y = X\\beta + Z\\gamma + \\epsilon

    where :math:`Z` are a set of regressors that are one of:

    * Powers of :math:`X\\hat{\\beta}` from the original regression.
    * Powers of :math:`X`, excluding the constant and binary regressors.
    * Powers of the first principal component of :math:`X`. If the
      model includes a constant, this column is dropped before computing
      the principal component. In either case, the principal component
      is extracted from the correlation matrix of remaining columns.

    The test is a Wald test of the null :math:`H_0:\\gamma=0`. If use_f
    is True, then the quadratic-form test statistic is divided by the
    number of restrictions and the F distribution is used to compute
    the critical value.
    """
    if not isinstance(res, RegressionResultsWrapper):
        raise TypeError('result must come from a linear regression model')
    if bool(res.model.k_constant) and res.model.exog.shape[1] == 1:
        raise ValueError('exog contains only a constant column. The RESET test requires exog to have at least 1 non-constant column.')
    test_type = string_like(test_type, 'test_type', options=('fitted', 'exog', 'princomp'))
    cov_kwargs = dict_like(cov_kwargs, 'cov_kwargs', optional=True)
    use_f = bool_like(use_f, 'use_f')
    if isinstance(power, int):
        if power < 2:
            raise ValueError('power must be >= 2')
        power = np.arange(2, power + 1, dtype=int)
    else:
        try:
            power = np.array(power, dtype=int)
        except Exception:
            raise ValueError('power must be an integer or list of integers')
        if power.ndim != 1 or len(set(power)) != power.shape[0] or (power < 2).any():
            raise ValueError('power must contains distinct integers all >= 2')
    exog = res.model.exog
    if test_type == 'fitted':
        aug = np.asarray(res.fittedvalues)[:, None]
    elif test_type == 'exog':
        aug = res.model.exog
        binary = (exog == exog.max(axis=0)) | (exog == exog.min(axis=0))
        binary = binary.all(axis=0)
        if binary.all():
            raise ValueError('Model contains only constant or binary data')
        aug = aug[:, ~binary]
    else:
        from statsmodels.multivariate.pca import PCA
        aug = exog
        if res.k_constant:
            retain = np.arange(aug.shape[1]).tolist()
            retain.pop(int(res.model.data.const_idx))
            aug = aug[:, retain]
        pca = PCA(aug, ncomp=1, standardize=bool(res.k_constant), demean=bool(res.k_constant), method='nipals')
        aug = pca.factors[:, :1]
    aug_exog = np.hstack([exog] + [aug ** p for p in power])
    mod_class = res.model.__class__
    mod = mod_class(res.model.data.endog, aug_exog)
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    res = mod.fit(cov_type=cov_type, cov_kwargs=cov_kwargs)
    nrestr = aug_exog.shape[1] - exog.shape[1]
    nparams = aug_exog.shape[1]
    r_mat = np.eye(nrestr, nparams, k=nparams - nrestr)
    return res.wald_test(r_mat, use_f=use_f, scalar=True)