import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_find_assign_op_dedent():
    """
    be careful that empty token like dedent are not counted as parens
    """

    class Tk:

        def __init__(self, s):
            self.string = s
    assert _find_assign_op([Tk(s) for s in ('', 'a', '=', 'b')]) == 2
    assert _find_assign_op([Tk(s) for s in ('', '(', 'a', '=', 'b', ')', '=', '5')]) == 6