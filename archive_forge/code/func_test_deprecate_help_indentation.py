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
@pytest.mark.parametrize('old_func, new_func', [(old_func4, new_func4), (old_func5, new_func5), (old_func6, new_func6)])
def test_deprecate_help_indentation(old_func, new_func):
    _compare_docs(old_func, new_func)
    for knd, func in (('old', old_func), ('new', new_func)):
        for li, line in enumerate(func.__doc__.split('\n')):
            if li == 0:
                assert line.startswith('    ') or not line.startswith(' '), knd
            elif line:
                assert line.startswith('    '), knd