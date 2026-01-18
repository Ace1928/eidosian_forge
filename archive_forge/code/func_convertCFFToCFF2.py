from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
def convertCFFToCFF2(self, otFont):
    """Converts this object from CFF format to CFF2 format. This conversion
        is done 'in-place'. The conversion cannot be reversed.

        This assumes a decompiled CFF table. (i.e. that the object has been
        filled via :meth:`decompile`.)"""
    self.major = 2
    cff2GetGlyphOrder = self.otFont.getGlyphOrder
    topDictData = TopDictIndex(None, cff2GetGlyphOrder)
    topDictData.items = self.topDictIndex.items
    self.topDictIndex = topDictData
    topDict = topDictData[0]
    if hasattr(topDict, 'Private'):
        privateDict = topDict.Private
    else:
        privateDict = None
    opOrder = buildOrder(topDictOperators2)
    topDict.order = opOrder
    topDict.cff2GetGlyphOrder = cff2GetGlyphOrder
    for entry in topDictOperators:
        key = entry[1]
        if key not in opOrder:
            if key in topDict.rawDict:
                del topDict.rawDict[key]
            if hasattr(topDict, key):
                delattr(topDict, key)
    if not hasattr(topDict, 'FDArray'):
        fdArray = topDict.FDArray = FDArrayIndex()
        fdArray.strings = None
        fdArray.GlobalSubrs = topDict.GlobalSubrs
        topDict.GlobalSubrs.fdArray = fdArray
        charStrings = topDict.CharStrings
        if charStrings.charStringsAreIndexed:
            charStrings.charStringsIndex.fdArray = fdArray
        else:
            charStrings.fdArray = fdArray
        fontDict = FontDict()
        fontDict.setCFF2(True)
        fdArray.append(fontDict)
        fontDict.Private = privateDict
        privateOpOrder = buildOrder(privateDictOperators2)
        for entry in privateDictOperators:
            key = entry[1]
            if key not in privateOpOrder:
                if key in privateDict.rawDict:
                    del privateDict.rawDict[key]
                if hasattr(privateDict, key):
                    delattr(privateDict, key)
    else:
        fdArray = topDict.FDArray
        privateOpOrder = buildOrder(privateDictOperators2)
        for fontDict in fdArray:
            fontDict.setCFF2(True)
            for key in fontDict.rawDict.keys():
                if key not in fontDict.order:
                    del fontDict.rawDict[key]
                    if hasattr(fontDict, key):
                        delattr(fontDict, key)
            privateDict = fontDict.Private
            for entry in privateDictOperators:
                key = entry[1]
                if key not in privateOpOrder:
                    if key in privateDict.rawDict:
                        del privateDict.rawDict[key]
                    if hasattr(privateDict, key):
                        delattr(privateDict, key)
    file = BytesIO()
    self.compile(file, otFont, isCFF2=True)
    file.seek(0)
    self.decompile(file, otFont, isCFF2=True)