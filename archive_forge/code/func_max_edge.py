import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def max_edge(g, src, dst, n):
    default = {'weight': -np.inf}
    w1 = g[n].get(src, default)['weight']
    w2 = g[n].get(dst, default)['weight']
    return {'weight': max(w1, w2)}