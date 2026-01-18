import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
def test_expected_warnings_noop():
    with expected_warnings(['Expected warnings test']):
        with expected_warnings(None):
            warn('Expected warnings test')