from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
@cached_data
def resid_working(self):
    """
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
        """
    val = self.resid_response * self.family.link.deriv(self.mu)
    val *= self._n_trials
    return val