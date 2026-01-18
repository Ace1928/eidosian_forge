import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_find_magic_escape():
    check_find(ipt2.EscapedCommand, MULTILINE_MAGIC)
    check_find(ipt2.EscapedCommand, INDENTED_MAGIC)
    check_find(ipt2.EscapedCommand, MULTILINE_MAGIC_ASSIGN, match=False)