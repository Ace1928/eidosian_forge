import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
def test_assert_raises_regex_context_manager():
    with assert_raises_regex(ValueError, 'no deprecation warning'):
        raise ValueError('no deprecation warning')