import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def check_find(transformer, case, match=True):
    sample, expected_start, _ = case
    tbl = make_tokens_by_line(sample)
    res = transformer.find(tbl)
    if match:
        assert (res.start_line + 1, res.start_col) == expected_start
        return res
    else:
        assert res is None