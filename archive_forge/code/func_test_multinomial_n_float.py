import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_multinomial_n_float(self):
    random.multinomial(100.5, [0.2, 0.8])