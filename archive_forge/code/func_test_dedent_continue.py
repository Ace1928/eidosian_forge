import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_dedent_continue(self):
    isp = self.isp
    isp.push('while 1:\n    continues = 5')
    self.assertEqual(isp.get_indent_spaces(), 4)
    isp.push('while 1:\n     continue')
    self.assertEqual(isp.get_indent_spaces(), 0)
    isp.push('while 1:\n     continue   ')
    self.assertEqual(isp.get_indent_spaces(), 0)