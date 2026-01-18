from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
def generate_formats(y):
    y = np.array(y)
    yield (y, 'array of ints')
    yield (y.tolist(), 'list of ints')
    yield ([str(x) + '-a' for x in y.tolist()], 'list of strs')
    yield (np.array([str(x) + '-a' for x in y.tolist()], dtype=object), 'array of strs')
    yield (y - 1, 'including negative ints')
    yield (y + 1, 'strictly positive ints')