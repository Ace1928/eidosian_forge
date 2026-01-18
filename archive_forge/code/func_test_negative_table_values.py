import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
def test_negative_table_values(self):
    with pytest.raises(ValueError, match='must be nonnegative'):
        odds_ratio([[1, 2], [3, -4]])