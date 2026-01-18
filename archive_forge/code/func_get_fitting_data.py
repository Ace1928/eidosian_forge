import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def get_fitting_data(self, vname):
    """
        Return the data needed to fit a model for imputation.

        The data is used to impute variable `vname`, and therefore
        only includes cases for which `vname` is observed.

        Values of type `PatsyFormula` in `init_kwds` or `fit_kwds` are
        processed through Patsy and subset to align with the model's
        endog and exog.

        Parameters
        ----------
        vname : str
           The variable for which the fitting data is returned.

        Returns
        -------
        endog : DataFrame
            Observed values of `vname`.
        exog : DataFrame
            Regression design matrix for imputing `vname`.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """
    ix = self.ix_obs[vname]
    formula = self.conditional_formula[vname]
    endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
    endog = np.require(endog.iloc[ix, 0], requirements='W')
    exog = np.require(exog.iloc[ix, :], requirements='W')
    init_kwds = self._process_kwds(self.init_kwds[vname], ix)
    fit_kwds = self._process_kwds(self.fit_kwds[vname], ix)
    return (endog, exog, init_kwds, fit_kwds)