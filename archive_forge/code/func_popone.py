import abc
import sys
import types
from collections.abc import Mapping, MutableMapping
@abc.abstractmethod
def popone(self, key, default=None):
    raise KeyError