import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_both_subclusters_updated():
    """Check that both subclusters are updated when a node a split, even when there are
    duplicated data points. Non-regression test for #23269.
    """
    X = np.array([[-2.6192791, -1.5053215], [-2.9993038, -1.6863596], [-2.3724914, -1.3438171], [-2.336792, -1.3417323], [-2.4089134, -1.3290224], [-2.3724914, -1.3438171], [-3.364009, -1.8846745], [-2.3724914, -1.3438171], [-2.617677, -1.5003285], [-2.2960556, -1.3260119], [-2.3724914, -1.3438171], [-2.5459878, -1.4533926], [-2.25979, -1.3003055], [-2.4089134, -1.3290224], [-2.3724914, -1.3438171], [-2.4089134, -1.3290224], [-2.5459878, -1.4533926], [-2.3724914, -1.3438171], [-2.9720619, -1.7058647], [-2.336792, -1.3417323], [-2.3724914, -1.3438171]], dtype=np.float32)
    Birch(branching_factor=5, threshold=1e-05, n_clusters=None).fit(X)