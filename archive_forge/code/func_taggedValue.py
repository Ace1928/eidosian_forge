import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
def taggedValue(key, value):
    """Attaches a tagged value to an interface at definition time."""
    f_locals = sys._getframe(1).f_locals
    tagged_values = f_locals.setdefault(TAGGED_DATA, {})
    tagged_values[key] = value
    return _decorator_non_return