import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_invalid_scalar(self):
    assert_raises(TypeError, random.RandomState, -0.5)
    assert_raises(ValueError, random.RandomState, -1)