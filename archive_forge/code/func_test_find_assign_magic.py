import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_find_assign_magic():
    check_find(ipt2.MagicAssign, MULTILINE_MAGIC_ASSIGN)
    check_find(ipt2.MagicAssign, MULTILINE_SYSTEM_ASSIGN, match=False)
    check_find(ipt2.MagicAssign, MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT, match=False)