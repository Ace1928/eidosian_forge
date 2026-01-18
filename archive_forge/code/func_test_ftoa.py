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
def test_ftoa(self):
    self.assertEqual(ftoa(10.0), '10')
    self.assertEqual(ftoa(1), '1')
    self.assertEqual(ftoa(1.0), '1')
    self.assertEqual(ftoa(-1.0), '-1')
    self.assertEqual(ftoa(0.0), '0')
    self.assertEqual(ftoa(1e+100), '1e+100')
    self.assertEqual(ftoa(1e-100), '1e-100')
    self.assertEqual(ftoa(10.0, True), '10')
    self.assertEqual(ftoa(1, True), '1')
    self.assertEqual(ftoa(1.0, True), '1')
    self.assertEqual(ftoa(-1.0, True), '(-1)')
    self.assertEqual(ftoa(0.0, True), '0')
    self.assertEqual(ftoa(1e+100, True), '1e+100')
    self.assertEqual(ftoa(1e-100, True), '1e-100')
    self.assertIsNone(ftoa(None))
    m = ConcreteModel()
    m.x = Var()
    with self.assertRaisesRegex(ValueError, 'Converting non-fixed bound or value to string: 2\\*x'):
        self.assertIsNone(ftoa(2 * m.x))