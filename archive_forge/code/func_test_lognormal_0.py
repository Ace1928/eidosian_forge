import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_lognormal_0(self):
    assert_equal(random.lognormal(sigma=0), 1)
    assert_raises(ValueError, random.lognormal, sigma=-0.0)