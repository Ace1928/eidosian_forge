import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def plot_scatter_all_pairs(self, title=None):
    from statsmodels.graphics.plot_grids import scatter_ellipse
    if self.model.k_exog_re < 2:
        raise ValueError('less than two variables available')
    return scatter_ellipse(self.params_random_units, ell_kwds={'color': 'r'})