import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
def getRegisteredConformers(self):
    """
        Return an iterable of the classes that are known to conform to
        the ABC this interface parallels.
        """
    based_on = self.__abc
    try:
        registered = list(based_on._abc_registry) + list(based_on._abc_cache)
    except AttributeError:
        from abc import _get_dump
        data = _get_dump(based_on)
        registry = data[0]
        cache = data[1]
        registered = [x() for x in itertools.chain(registry, cache)]
        registered = [x for x in registered if x is not None]
    return set(itertools.chain(registered, self.__extra_classes))