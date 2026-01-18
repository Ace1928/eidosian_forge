import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
@pytest.mark.parametrize('kind', ['sample', 'conditional'])
@pytest.mark.parametrize('bad_table', [123, 'foo', [10, 11, 12]])
def test_invalid_table_shape(self, kind, bad_table):
    with pytest.raises(ValueError, match='Invalid shape'):
        odds_ratio(bad_table, kind=kind)