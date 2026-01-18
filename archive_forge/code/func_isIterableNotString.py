from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
def isIterableNotString(o):
    from collections import abc
    return isinstance(o, abc.Iterable) and (not isinstance(o, memoryview)) and (not hasattr(o, 'isalpha'))