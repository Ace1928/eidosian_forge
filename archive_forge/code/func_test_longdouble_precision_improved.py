import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_longdouble_precision_improved():
    if os.name != 'nt':
        assert not longdouble_precision_improved()