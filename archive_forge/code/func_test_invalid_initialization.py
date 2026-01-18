import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_invalid_initialization(self):
    assert_raises(ValueError, random.RandomState, MT19937)