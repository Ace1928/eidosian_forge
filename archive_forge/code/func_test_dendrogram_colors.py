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
def test_dendrogram_colors(self, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
    set_link_color_palette(['c', 'm', 'y', 'k'])
    R = dendrogram(Z, no_plot=True, above_threshold_color='g', color_threshold=250)
    set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])
    color_list = R['color_list']
    assert_equal(color_list, ['c', 'm', 'g', 'g', 'g'])
    set_link_color_palette(None)