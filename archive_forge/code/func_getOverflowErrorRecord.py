from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def getOverflowErrorRecord(self, item):
    LookupListIndex = SubTableIndex = itemName = itemIndex = None
    if self.name == 'LookupList':
        LookupListIndex = item.repeatIndex
    elif self.name == 'Lookup':
        LookupListIndex = self.repeatIndex
        SubTableIndex = item.repeatIndex
    else:
        itemName = getattr(item, 'name', '<none>')
        if hasattr(item, 'repeatIndex'):
            itemIndex = item.repeatIndex
        if self.name == 'SubTable':
            LookupListIndex = self.parent.repeatIndex
            SubTableIndex = self.repeatIndex
        elif self.name == 'ExtSubTable':
            LookupListIndex = self.parent.parent.repeatIndex
            SubTableIndex = self.parent.repeatIndex
        else:
            itemName = '.'.join([self.name, itemName])
            p1 = self.parent
            while p1 and p1.name not in ['ExtSubTable', 'SubTable']:
                itemName = '.'.join([p1.name, itemName])
                p1 = p1.parent
            if p1:
                if p1.name == 'ExtSubTable':
                    LookupListIndex = p1.parent.parent.repeatIndex
                    SubTableIndex = p1.parent.repeatIndex
                else:
                    LookupListIndex = p1.parent.repeatIndex
                    SubTableIndex = p1.repeatIndex
    return OverflowErrorRecord((self.tableTag, LookupListIndex, SubTableIndex, itemName, itemIndex))