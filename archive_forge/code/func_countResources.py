from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def countResources(self, resType):
    """Return the number of resources of a given type."""
    try:
        return len(self[resType])
    except KeyError:
        return 0