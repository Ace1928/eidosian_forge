import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
@skip_if_array_api_gpu
@array_api_compatible
@pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
def test_dendrogram_plot(self, xp):
    for orientation in ['top', 'bottom', 'left', 'right']:
        self.check_dendrogram_plot(orientation, xp)