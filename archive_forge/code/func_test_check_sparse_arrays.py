import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_check_sparse_arrays(csr_container):
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_sparse = csr_container(XA)
    XB = rng.random_sample((5, 4))
    XB_sparse = csr_container(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_sparse, XB_sparse)
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XB_checked)
    assert abs(XB_sparse - XB_checked).sum() == 0
    XA_checked, XA_2_checked = check_pairwise_arrays(XA_sparse, XA_sparse)
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XA_2_checked)
    assert abs(XA_2_checked - XA_checked).sum() == 0