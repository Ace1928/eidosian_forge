import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def assert_equal_w_dt(a, b, err_msg):
    assert_equal(a.dtype, b.dtype, err_msg=err_msg)
    assert_equal(a, b, err_msg=err_msg)