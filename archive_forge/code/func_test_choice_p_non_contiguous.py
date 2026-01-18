import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_p_non_contiguous(self):
    p = np.ones(10) / 5
    p[1::2] = 3.0
    random.seed(self.seed)
    non_contig = random.choice(5, 3, p=p[::2])
    random.seed(self.seed)
    contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
    assert_array_equal(non_contig, contig)