import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_kbinsdiscretizer_subsample_default():
    X = np.array([-2, 1.5, -4, -1]).reshape(-1, 1)
    kbd_default = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    kbd_default.fit(X)
    kbd_without_subsampling = clone(kbd_default)
    kbd_without_subsampling.set_params(subsample=None)
    kbd_without_subsampling.fit(X)
    for bin_kbd_default, bin_kbd_with_subsampling in zip(kbd_default.bin_edges_[0], kbd_without_subsampling.bin_edges_[0]):
        np.testing.assert_allclose(bin_kbd_default, bin_kbd_with_subsampling)
    assert kbd_default.bin_edges_.shape == kbd_without_subsampling.bin_edges_.shape