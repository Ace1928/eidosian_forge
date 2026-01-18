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
def test_filedeterminism(self):
    with LoggingIntercept() as LOG:
        a = FileDeterminism(10)
    self.assertEqual(a, FileDeterminism.ORDERED)
    self.assertEqual('', LOG.getvalue())
    self.assertEqual(str(a), 'FileDeterminism.ORDERED')
    self.assertEqual(f'{a}', 'FileDeterminism.ORDERED')
    with LoggingIntercept() as LOG:
        a = FileDeterminism(1)
    self.assertEqual(a, FileDeterminism.SORT_INDICES)
    self.assertIn('FileDeterminism(1) is deprecated.  Please use FileDeterminism.SORT_INDICES (20)', LOG.getvalue().replace('\n', ' '))
    with self.assertRaisesRegex(ValueError, '5 is not a valid FileDeterminism'):
        FileDeterminism(5)