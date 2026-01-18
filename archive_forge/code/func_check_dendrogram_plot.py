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
def check_dendrogram_plot(self, orientation, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
    expected = {'color_list': ['C1', 'C0', 'C0', 'C0', 'C0'], 'dcoord': [[0.0, 138.0, 138.0, 0.0], [0.0, 219.0, 219.0, 0.0], [0.0, 255.0, 255.0, 219.0], [0.0, 268.0, 268.0, 255.0], [138.0, 295.0, 295.0, 268.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0], [45.0, 45.0, 55.0, 55.0], [35.0, 35.0, 50.0, 50.0], [25.0, 25.0, 42.5, 42.5], [10.0, 10.0, 33.75, 33.75]], 'ivl': ['2', '5', '1', '0', '3', '4'], 'leaves': [2, 5, 1, 0, 3, 4], 'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0', 'C0']}
    fig = plt.figure()
    ax = fig.add_subplot(221)
    R1 = dendrogram(Z, ax=ax, orientation=orientation)
    R1['dcoord'] = np.asarray(R1['dcoord'])
    assert_equal(R1, expected)
    dendrogram(Z, ax=ax, orientation=orientation, leaf_font_size=20, leaf_rotation=90)
    testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
    assert_equal(testlabel.get_rotation(), 90)
    assert_equal(testlabel.get_size(), 20)
    dendrogram(Z, ax=ax, orientation=orientation, leaf_rotation=90)
    testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
    assert_equal(testlabel.get_rotation(), 90)
    dendrogram(Z, ax=ax, orientation=orientation, leaf_font_size=20)
    testlabel = ax.get_xticklabels()[0] if orientation in ['top', 'bottom'] else ax.get_yticklabels()[0]
    assert_equal(testlabel.get_size(), 20)
    plt.close()
    R2 = dendrogram(Z, orientation=orientation)
    plt.close()
    R2['dcoord'] = np.asarray(R2['dcoord'])
    assert_equal(R2, expected)