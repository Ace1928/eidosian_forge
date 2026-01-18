import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
import statsmodels.genmod.families as F
from statsmodels.genmod.families.family import Tweedie
import statsmodels.genmod.families.links as L
@pytest.mark.parametrize('family, links', link_cases)
def test_invalid_family_link(family, links):
    invalid_links = all_links - links
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            msg = 'Negative binomial dispersion parameter alpha not set. Using default value alpha=1.0.'
            warnings.filterwarnings('ignore', message=msg, category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            for link in invalid_links:
                family(link())