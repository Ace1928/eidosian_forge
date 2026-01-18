import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_find_next_indent():
    for code, exp in indentation_samples:
        res = isp.find_next_indent(code)
        msg = '{!r} != {!r} (expected)\n Code: {!r}'.format(res, exp, code)
        assert res == exp, msg