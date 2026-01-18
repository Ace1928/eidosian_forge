from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def cov_ybar(self):
    """Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \\sqrt(T) (\\bar{y} - \\mu) \\rightarrow
                  {\\cal N}(0, \\Sigma_{\\bar{y}}) \\\\

            \\Sigma_{\\bar{y}} = B \\Sigma_u B^\\prime, \\text{where }
                  B = (I_K - A_1 - \\cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """
    Ainv = np.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
    return Ainv @ self.sigma_u @ Ainv.T