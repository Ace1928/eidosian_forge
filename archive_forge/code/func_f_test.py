from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def f_test(self, r_matrix, cov_p=None, invcov=None):
    """
        Compute the F-test for a joint linear hypothesis.

        This is a special case of `wald_test` that always uses the F
        distribution.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length k row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        t_test : Perform a single hypothesis test.
        wald_test : Perform a Wald-test using a quadratic form.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print(results.f_test(A))
        <F test: F=array([[ 330.28533923]]), p=4.984030528700946e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.fvalue
        330.2853392346658
        >>> results.f_pvalue
        4.98403096572e-10

        >>> B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print(results.f_test(B))
        <F test: F=array([[ 9.74046187]]), p=0.005605288531708235, df_denom=9, df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.datasets import longley
        >>> from statsmodels.formula.api import ols
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print(f_test)
        <F test: F=array([[ 144.17976065]]), p=6.322026217355609e-08, df_denom=9, df_num=3>
        """
    res = self.wald_test(r_matrix, cov_p=cov_p, invcov=invcov, use_f=True, scalar=True)
    return res