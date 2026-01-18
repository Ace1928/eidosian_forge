from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
@staticmethod
def openDataFork(path):
    with open(path, 'rb') as datafork:
        data = datafork.read()
    infile = BytesIO(data)
    infile.name = path
    return infile