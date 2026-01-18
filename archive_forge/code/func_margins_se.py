from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def margins_se(self):
    raise NotImplementedError