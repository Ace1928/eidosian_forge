import inspect
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
import pyomo.common.tests.dep_mod as dep_mod
from . import deps
def test_UnavailableClass(self):
    module_obj, module_available = attempt_import('__there_is_no_module_named_this__', 'Testing import of a non-existent module', defer_check=False)

    class A_Class(UnavailableClass(module_obj)):
        pass
    with self.assertRaisesRegex(DeferredImportError, "The class 'A_Class' cannot be created because a needed optional dependency was not found \\(import raised ModuleNotFoundError: No module named '__there_is_no_module_named_this__'\\)"):
        A_Class()
    with self.assertRaisesRegex(DeferredImportError, "The class attribute 'A_Class.method' is not available because a needed optional dependency was not found \\(import raised ModuleNotFoundError: No module named '__there_is_no_module_named_this__'\\)"):
        A_Class.method()