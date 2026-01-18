import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def empty_to_1(x):
    assert_(len(x) == 0)
    return 1