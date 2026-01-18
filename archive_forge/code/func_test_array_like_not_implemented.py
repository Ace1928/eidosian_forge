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
def test_array_like_not_implemented(self):
    self.add_method('array', self.MyArray)
    ref = self.MyArray.array()
    with assert_raises_regex(TypeError, 'no implementation found'):
        array_like = np.asarray(1, like=ref)