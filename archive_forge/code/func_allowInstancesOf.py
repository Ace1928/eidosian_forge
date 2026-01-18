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
def allowInstancesOf(self, *classes):
    """
        SecurityOptions.allowInstances(klass, klass, ...): allow instances
        of the specified classes

        This will also allow the 'instance', 'class' (renamed 'classobj' in
        Python 2.3), and 'module' types, as well as basic types.
        """
    self.allowBasicTypes()
    self.allowTypes('instance', 'class', 'classobj', 'module')
    for klass in classes:
        self.allowTypes(qual(klass))
        self.allowModules(klass.__module__)
        self.allowedClasses[klass] = 1