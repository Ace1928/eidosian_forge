import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
@pytest.mark.parametrize('code, expected, number', examples)
def test_check_complete_param(code, expected, number):
    cc = ipt2.TransformerManager().check_complete
    assert cc(code) == (expected, number)