import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_cellmagic_help(self):
    self.sp.push('%%cellm?')
    assert self.sp.push_accepts_more() is False