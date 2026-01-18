from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def setEntry(self, tag, entry):
    if tag in self.tables:
        raise TTLibError("cannot rewrite '%s' table" % tag)
    self.tables[tag] = entry