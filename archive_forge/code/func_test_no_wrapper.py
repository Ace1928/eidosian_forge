import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_no_wrapper(self):
    array = np.array(1)
    func = lambda x: x
    with assert_raises_regex(AttributeError, '_implementation'):
        array.__array_function__(func=func, types=(np.ndarray,), args=(array,), kwargs={})