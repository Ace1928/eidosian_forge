import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_dedent_break(self):
    isp = self.isp
    isp.push('while 1:\n    breaks = 5')
    self.assertEqual(isp.get_indent_spaces(), 4)
    isp.push('while 1:\n     break')
    self.assertEqual(isp.get_indent_spaces(), 0)
    isp.push('while 1:\n     break   ')
    self.assertEqual(isp.get_indent_spaces(), 0)