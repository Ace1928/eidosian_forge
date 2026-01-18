import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
@pytest.mark.xfail(platform.python_implementation() == 'PyPy', reason='fail on pypy')
@pytest.mark.parametrize('value, expected', [('def foo():\n    """', ('incomplete', 4)), ('async with example:\n    pass', ('incomplete', 4)), ('async with example:\n    pass\n    ', ('complete', None))])
def test_check_complete_II(value, expected):
    """
    Test that multiple line strings are properly handled.

    Separate test function for convenience

    """
    cc = ipt2.TransformerManager().check_complete
    assert cc(value) == expected