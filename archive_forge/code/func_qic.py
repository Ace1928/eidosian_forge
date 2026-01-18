from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def qic(self, scale=None, n_step=1000):
    """
        Returns the QIC and QICu information criteria.

        See GEE.qic for documentation.
        """
    if scale is None:
        warnings.warn('QIC values obtained using scale=None are not appropriate for comparing models')
    if scale is None:
        scale = self.scale
    _, qic, qicu = self.model.qic(self.params, scale, self.cov_params(), n_step=n_step)
    return (qic, qicu)