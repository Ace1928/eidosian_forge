import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def pseudo_input(lines):
    """Return a function that acts like raw_input but feeds the input list."""
    ilines = iter(lines)

    def raw_in(prompt):
        try:
            return next(ilines)
        except StopIteration:
            return ''
    return raw_in