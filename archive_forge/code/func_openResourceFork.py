from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
@staticmethod
def openResourceFork(path):
    if hasattr(path, '__fspath__'):
        path = path.__fspath__()
    with open(path + '/..namedfork/rsrc', 'rb') as resfork:
        data = resfork.read()
    infile = BytesIO(data)
    infile.name = path
    return infile