import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def gen_random(state, out):
    out[...] = state.multinomial(10, [1 / 6.0] * 6, size=10000)