from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def getIndices(self, resType):
    """Returns a list of indices of resources of a given type."""
    numRes = self.countResources(resType)
    if numRes:
        return list(range(1, numRes + 1))
    else:
        return []