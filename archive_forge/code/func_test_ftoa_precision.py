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
@unittest.skipIf(not numpy_available, 'NumPy is not available')
def test_ftoa_precision(self):
    log = StringIO()
    with LoggingIntercept(log, 'pyomo.core', logging.WARNING):
        f = np.longdouble('1.1234567890123456789')
        a = ftoa(f)
    self.assertEqual(a, '1.1234567890123457')
    if f == float(f):
        test = self.assertNotRegex
    else:
        test = self.assertRegex
    test(log.getvalue(), '.*Converting 1.1234567890123456789 to string resulted in loss of precision')