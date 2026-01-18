from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.fixture(params=range(MDATA_COUNT))
def mdata_x(request, reference_data):
    return reference_data['X'][request.param]