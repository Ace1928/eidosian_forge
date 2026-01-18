import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_named_argument_initialization(self):
    rs1 = np.random.RandomState(123456789)
    rs2 = np.random.RandomState(seed=123456789)
    assert rs1.randint(0, 100) == rs2.randint(0, 100)