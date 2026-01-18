import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def build_operands(self, string, size_dict=global_size_dict):
    operands = [string]
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [size_dict[x] for x in term]
        operands.append(np.random.rand(*dims))
    return operands