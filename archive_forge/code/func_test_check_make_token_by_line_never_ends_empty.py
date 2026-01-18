import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_check_make_token_by_line_never_ends_empty():
    """
    Check that not sequence of single or double characters ends up leading to en empty list of tokens
    """
    from string import printable
    for c in printable:
        assert make_tokens_by_line(c)[-1] != []
        for k in printable:
            assert make_tokens_by_line(c + k)[-1] != []