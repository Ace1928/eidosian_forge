import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
@pytest.mark.skipif(sys.flags.optimize == 2, reason='-OO discards docstrings')
def test_deprecate_preserve_whitespace():
    assert_('\n        Bizarre' in new_func5.__doc__)