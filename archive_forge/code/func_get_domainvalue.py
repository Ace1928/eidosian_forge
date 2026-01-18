import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
def get_domainvalue(link):
    """
    Get a value in the domain for a given family.
    """
    z = -np.log(np.random.uniform(0, 1))
    if isinstance(link, links.CLogLog):
        z = min(z, 3)
    elif isinstance(link, links.LogLog):
        z = max(z, -3)
    elif isinstance(link, (links.NegativeBinomial, links.LogC)):
        z = -z
    return z