import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_import_exceptions(self):
    mod, avail = attempt_import('pyomo.common.tests.dep_mod_except', defer_check=True, only_catch_importerror=True)
    with self.assertRaisesRegex(ValueError, 'cannot import module'):
        bool(avail)
    self.assertFalse(avail)
    mod, avail = attempt_import('pyomo.common.tests.dep_mod_except', defer_check=True, only_catch_importerror=False)
    self.assertFalse(avail)
    self.assertFalse(avail)
    mod, avail = attempt_import('pyomo.common.tests.dep_mod_except', defer_check=True, catch_exceptions=(ImportError, ValueError))
    self.assertFalse(avail)
    self.assertFalse(avail)
    with self.assertRaisesRegex(ValueError, 'Cannot specify both only_catch_importerror and catch_exceptions'):
        mod, avail = attempt_import('pyomo.common.tests.dep_mod_except', defer_check=True, only_catch_importerror=True, catch_exceptions=(ImportError,))