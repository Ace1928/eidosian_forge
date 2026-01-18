import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_check_complete(self):
    isp = self.isp
    self.assertEqual(isp.check_complete('a = 1'), ('complete', None))
    self.assertEqual(isp.check_complete('for a in range(5):'), ('incomplete', 4))
    self.assertEqual(isp.check_complete('raise = 2'), ('invalid', None))
    self.assertEqual(isp.check_complete('a = [1,\n2,'), ('incomplete', 0))
    self.assertEqual(isp.check_complete('def a():\n x=1\n global x'), ('invalid', None))