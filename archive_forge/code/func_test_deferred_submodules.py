import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_deferred_submodules(self):
    import pyomo
    pyo_ver = pyomo.version.version
    self.assertIsInstance(deps.pyo, DeferredImportModule)
    self.assertIsNone(deps.pyo._submodule_name)
    self.assertEqual(deps.pyo_available._deferred_submodules, ['.version', '.common', '.common.tests', '.common.tests.dep_mod'])
    version = deps.pyo.version
    self.assertIsInstance(deps.pyo, DeferredImportModule)
    self.assertIsNone(deps.pyo._submodule_name)
    self.assertIsInstance(deps.dm, DeferredImportModule)
    self.assertEqual(deps.dm._submodule_name, '.common.tests.dep_mod')
    self.assertIsInstance(version, DeferredImportModule)
    self.assertEqual(version._submodule_name, '.version')
    self.assertEqual(version.version, pyo_ver)
    self.assertTrue(inspect.ismodule(deps.pyo))
    self.assertTrue(inspect.ismodule(deps.dm))
    with self.assertRaisesRegex(ValueError, 'deferred_submodules is only valid if defer_check==True'):
        mod, mod_available = attempt_import('nonexisting.module', defer_check=False, deferred_submodules={'submod': None})
    mod, mod_available = attempt_import('nonexisting.module', defer_check=True, deferred_submodules={'submod.subsubmod': None})
    self.assertIs(type(mod), DeferredImportModule)
    self.assertFalse(mod_available)
    _mod = mod_available._module
    self.assertIs(type(_mod), ModuleUnavailable)
    self.assertTrue(hasattr(_mod, 'submod'))
    self.assertIs(type(_mod.submod), ModuleUnavailable)
    self.assertTrue(hasattr(_mod.submod, 'subsubmod'))
    self.assertIs(type(_mod.submod.subsubmod), ModuleUnavailable)