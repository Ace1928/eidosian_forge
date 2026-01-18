from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
def check_implements(self, cls, interfaces, error_mode):
    """ Checks that the class implements the specified interfaces.

            'interfaces' can be a single interface or a list of interfaces.
        """
    try:
        iter(interfaces)
    except TypeError:
        interfaces = [interfaces]
    if issubclass(cls, HasTraits):
        for interface in interfaces:
            if not self._check_has_traits_class(cls, interface, error_mode):
                return False
    else:
        for interface in interfaces:
            if not self._check_non_has_traits_class(cls, interface, error_mode):
                return False
    return True