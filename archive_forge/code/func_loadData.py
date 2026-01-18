from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def loadData(self, file):
    file.seek(self.offset)
    data = file.read(self.length)
    assert len(data) == self.length
    if hasattr(self.__class__, 'decodeData'):
        data = self.decodeData(data)
    return data