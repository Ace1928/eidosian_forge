import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal
def test_matmul_scalar():
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100)
    x1 = np.array(x1)
    with pytest.raises(ValueError):
        x1 @ x2