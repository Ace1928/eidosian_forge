from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
@cache_readonly
def hatmatrix_diag(self):
    """diagonal of hat matrix

        diag(X' xpxi X)

        where xpxi = (X'X + sigma2_e * lambd * sigma_prior)^{-1}

        Notes
        -----

        uses wexog, so this includes weights or sigma - check this case

        not clear whether I need to multiply by sigmahalf, i.e.

        (W^{-0.5} X) (X' W X)^{-1} (W^{-0.5} X)'  or
        (W X) (X' W X)^{-1} (W X)'

        projection y_hat = H y    or in terms of transformed variables (W^{-0.5} y)

        might be wrong for WLS and GLS case
        """
    xpxi = self.model.normalized_cov_params
    return (self.model.wexog * np.dot(xpxi, self.model.wexog.T).T).sum(1)