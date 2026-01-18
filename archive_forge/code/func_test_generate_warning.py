import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_generate_warning(self):
    mod, avail = attempt_import('pyomo.common.tests.dep_mod_except', defer_check=True, only_catch_importerror=False)
    log = StringIO()
    dep = StringIO()
    with LoggingIntercept(dep, 'pyomo.common.tests'):
        with LoggingIntercept(log, 'pyomo.common'):
            mod.generate_import_warning()
    self.assertIn('The pyomo.common.tests.dep_mod_except module (an optional Pyomo dependency) failed to import', log.getvalue())
    self.assertIn('DEPRECATED: use :py:class:`log_import_warning()`', dep.getvalue())
    log = StringIO()
    dep = StringIO()
    with LoggingIntercept(dep, 'pyomo'):
        with LoggingIntercept(log, 'pyomo.core.base'):
            mod.generate_import_warning('pyomo.core.base')
    self.assertIn('The pyomo.common.tests.dep_mod_except module (an optional Pyomo dependency) failed to import', log.getvalue())
    self.assertIn('DEPRECATED: use :py:class:`log_import_warning()`', dep.getvalue())