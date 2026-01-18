import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def allowModules(self, *modules):
    """
        SecurityOptions.allowModules(module, module, ...): allow modules by
        name. This will also allow the 'module' type.
        """
    for module in modules:
        if type(module) == types.ModuleType:
            module = module.__name__
        if not isinstance(module, bytes):
            module = module.encode('utf-8')
        self.allowedModules[module] = 1