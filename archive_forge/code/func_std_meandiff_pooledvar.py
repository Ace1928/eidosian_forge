import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def std_meandiff_pooledvar(self):
    """variance assuming equal variance in both data sets

        """
    d1 = self.d1
    d2 = self.d2
    var_pooled = (d1.sumsquares + d2.sumsquares) / (d1.nobs - 1 + d2.nobs - 1)
    return np.sqrt(var_pooled * (1.0 / d1.nobs + 1.0 / d2.nobs))