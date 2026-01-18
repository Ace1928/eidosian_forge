import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
def getversion(md):
    if not isinstance(md, _Base):
        raise TypeError('Parameter should be multidict or proxy')
    return md._impl._version