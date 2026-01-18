import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
@pytest.mark.parametrize('value, expected', [(')', ('invalid', None)), (']', ('invalid', None)), ('}', ('invalid', None)), (')(', ('invalid', None)), ('][', ('invalid', None)), ('}{', ('invalid', None)), (']()(', ('invalid', None)), ('())(', ('invalid', None)), (')[](', ('invalid', None)), ('()](', ('invalid', None))])
def test_check_complete_invalidates_sunken_brackets(value, expected):
    """
    Test that a single line with more closing brackets than the opening ones is
    interpreted as invalid
    """
    cc = ipt2.TransformerManager().check_complete
    assert cc(value) == expected