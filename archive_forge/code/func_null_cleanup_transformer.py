import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def null_cleanup_transformer(lines):
    """
    A cleanup transform that returns an empty list.
    """
    return []