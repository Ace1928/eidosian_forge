import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_cdist_dtype_equivalence(self, metric):
    eps = 1e-07
    tests = [(eo['random-bool-data'], self.valid_upcasts['bool']), (eo['random-uint-data'], self.valid_upcasts['uint']), (eo['random-int-data'], self.valid_upcasts['int']), (eo['random-float32-data'], self.valid_upcasts['float32'])]
    for test in tests:
        X1 = test[0][::5, ::-2]
        X2 = test[0][1::5, ::2]
        try:
            y1 = cdist(X1, X2, metric=metric)
        except Exception as e:
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            for new_type in test[1]:
                X1new = new_type(X1)
                X2new = new_type(X2)
                with pytest.raises(e_cls):
                    cdist(X1new, X2new, metric=metric)
        else:
            for new_type in test[1]:
                y2 = cdist(new_type(X1), new_type(X2), metric=metric)
                assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)