import sys
import weakref
from types import FunctionType
from types import MethodType
from types import ModuleType
from zope.interface._compat import _use_c_impl
from zope.interface.interface import Interface
from zope.interface.interface import InterfaceClass
from zope.interface.interface import NameAndModuleComparisonMixin
from zope.interface.interface import Specification
from zope.interface.interface import SpecificationBase
@_use_c_impl
def getObjectSpecification(ob):
    try:
        provides = ob.__provides__
    except AttributeError:
        provides = None
    if provides is not None:
        if isinstance(provides, SpecificationBase):
            return provides
    try:
        cls = ob.__class__
    except AttributeError:
        return _empty
    return implementedBy(cls)