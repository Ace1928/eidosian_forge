import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize(('r_plot', 'end'), [[[1, 2, 2.1, 2.2, 4, 8, 8, np.inf], 6], [[1, 2, 2.1, 2.2, 2.3, 4, 8, 8, np.inf], 0], [[1, 2, 2.1, 2, np.inf], 0], [[1, 2, 2.1, np.inf], 2]])
def test_extend_upward(r_plot, end):
    r_plot = np.array(r_plot)
    ratio = r_plot[:-1] / r_plot[1:]
    steep_upward = ratio <= 0.9
    downward = ratio > 1
    e = _extend_region(steep_upward, downward, 0, 2)
    assert e == end