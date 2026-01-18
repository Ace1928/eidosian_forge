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
def setUnjellyableForClass(classname, unjellyable):
    """
    Set which local class will represent a remote type.

    If you have written a Copyable class that you expect your client to be
    receiving, write a local "copy" class to represent it, then call::

        jellier.setUnjellyableForClass('module.package.Class', MyCopier).

    Call this at the module level immediately after its class
    definition. MyCopier should be a subclass of RemoteCopy.

    The classname may be a special tag returned by
    'Copyable.getTypeToCopyFor' rather than an actual classname.

    This call is also for cached classes, since there will be no
    overlap.  The rules are the same.
    """
    global unjellyableRegistry
    classname = _maybeClass(classname)
    unjellyableRegistry[classname] = unjellyable
    globalSecurity.allowTypes(classname)