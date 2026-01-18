import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_imported_deferred_import(self):
    self.assertIs(type(deps.has_bogus_nem), DeferredImportIndicator)
    self.assertIs(type(deps.bogus_nem), DeferredImportModule)
    with self.assertRaisesRegex(DeferredImportError, 'The bogus_nonexisting_module module \\(an optional Pyomo dependency\\) failed to import'):
        deps.test_access_bogus_hello()
    self.assertIs(deps.has_bogus_nem, False)
    self.assertIs(type(deps.bogus_nem), ModuleUnavailable)
    self.assertIs(dep_mod.bogus_nonexisting_module_available, False)
    self.assertIs(type(dep_mod.bogus_nonexisting_module), ModuleUnavailable)