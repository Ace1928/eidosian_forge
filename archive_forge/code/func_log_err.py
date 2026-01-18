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
def log_err(*args):
    self.called += 1
    extobj_err = args
    assert_(len(extobj_err) == 2)
    assert_('divide' in extobj_err[0])