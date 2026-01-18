import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_find_help():
    for case in [SIMPLE_HELP, DETAILED_HELP, MAGIC_HELP, HELP_IN_EXPR]:
        check_find(ipt2.HelpEnd, case)
    tf = check_find(ipt2.HelpEnd, HELP_CONTINUED_LINE)
    assert tf.q_line == 1
    assert tf.q_col == 3
    tf = check_find(ipt2.HelpEnd, HELP_MULTILINE)
    assert tf.q_line == 1
    assert tf.q_col == 8
    check_find(ipt2.HelpEnd, (['foo # bar?\n'], None, None), match=False)
    check_find(ipt2.HelpEnd, (["foo = '''bar?\n"], None, None), match=False)