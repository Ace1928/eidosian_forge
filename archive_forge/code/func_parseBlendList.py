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
def parseBlendList(s):
    valueList = []
    for element in s:
        if isinstance(element, str):
            continue
        name, attrs, content = element
        blendList = attrs['value'].split()
        blendList = [eval(val) for val in blendList]
        valueList.append(blendList)
    if len(valueList) == 1:
        valueList = valueList[0]
    return valueList