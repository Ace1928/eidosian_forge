import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_cannot_seed(self):
    rs = random.RandomState(PCG64(0))
    with assert_raises(TypeError):
        rs.seed(1234)