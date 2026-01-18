import numpy as np
from statsmodels.tools.tools import Bunch
def partial_project(endog, exog):
    """helper function to get linear projection or partialling out of variables

    endog variables are projected on exog variables

    Parameters
    ----------
    endog : ndarray
        array of variables where the effect of exog is partialled out.
    exog : ndarray
        array of variables on which the endog variables are projected.

    Returns
    -------
    res : instance of Bunch with

        - params : OLS parameter estimates from projection of endog on exog
        - fittedvalues : predicted values of endog given exog
        - resid : residual of the regression, values of endog with effect of
          exog partialled out

    Notes
    -----
    This is no-frills mainly for internal calculations, no error checking or
    array conversion is performed, at least for now.

    """
    x1, x2 = (endog, exog)
    params = np.linalg.pinv(x2).dot(x1)
    predicted = x2.dot(params)
    residual = x1 - predicted
    res = Bunch(params=params, fittedvalues=predicted, resid=residual)
    return res