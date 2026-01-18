import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_broadcast_size_error():
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))