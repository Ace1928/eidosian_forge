from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def plot_data(self, with_presample=False):
    """
        Plot the input time series.

        Parameters
        ----------
        with_presample : bool, default: `False`
            If `False`, the pre-sample data (the first `k_ar` values) will
            not be plotted.
        """
    y = self.y_all if with_presample else self.y_all[:, self.k_ar:]
    names = self.names
    dates = self.dates if with_presample else self.dates[self.k_ar:]
    plot.plot_mts(y.T, names=names, index=dates)