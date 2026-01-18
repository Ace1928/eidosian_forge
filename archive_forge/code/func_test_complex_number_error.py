import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_complex_number_error(self):

    class Visitor(object):
        pass
    visitor = Visitor()
    m = ConcreteModel()
    m.x = Var()
    exp = m.x ** (1 / 2)
    _halt = pyomo.repn.util.HALT_ON_EVALUATION_ERROR
    try:
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = True
        with LoggingIntercept() as LOG:
            with self.assertRaisesRegex(InvalidValueError, 'Pyomo Visitor does not support complex numbers'):
                complex_number_error(1j, visitor, exp)
        self.assertEqual(LOG.getvalue(), 'Complex number returned from expression\n\tmessage: Pyomo Visitor does not support complex numbers\n\texpression: x**0.5\n')
        with LoggingIntercept() as LOG:
            with self.assertRaisesRegex(InvalidValueError, 'Pyomo Visitor does not support complex numbers'):
                complex_number_error(1j, visitor, exp, "'(-1)**(0.5)'")
        self.assertEqual(LOG.getvalue(), "Complex number returned from expression '(-1)**(0.5)'\n\tmessage: Pyomo Visitor does not support complex numbers\n\texpression: x**0.5\n")
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = False
        with LoggingIntercept() as LOG:
            val = complex_number_error(1j, visitor, exp)
            self.assertEqual(str(val), 'InvalidNumber(1j)')
        self.assertEqual(LOG.getvalue(), 'Complex number returned from expression\n\tmessage: Pyomo Visitor does not support complex numbers\n\texpression: x**0.5\n')
    finally:
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = _halt