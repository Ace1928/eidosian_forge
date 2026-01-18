import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
def test_fastica_nowhiten():
    m = [[0, 1], [1, 0]]
    ica = FastICA(n_components=1, whiten=False, random_state=0)
    warn_msg = 'Ignoring n_components with whiten=False.'
    with pytest.warns(UserWarning, match=warn_msg):
        ica.fit(m)
    assert hasattr(ica, 'mixing_')