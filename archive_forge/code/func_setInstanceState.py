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
def setInstanceState(inst, unjellier, jellyList):
    """
    Utility method to default to 'normal' state rules in unserialization.
    """
    state = unjellier.unjelly(jellyList[1])
    if hasattr(inst, '__setstate__'):
        inst.__setstate__(state)
    else:
        inst.__dict__ = state
    return inst