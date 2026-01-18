from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@Appender(remove_parameters(arma_impulse_response.__doc__, ['ar', 'ma']))
def impulse_response(self, leads=None):
    leads = leads or self.nobs
    return arma_impulse_response(self.ar, self.ma, leads=leads)