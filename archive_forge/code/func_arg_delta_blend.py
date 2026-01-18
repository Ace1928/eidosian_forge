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
def arg_delta_blend(self, value):
    """A delta list with blend lists has to be *all* blend lists.

        The value is a list is arranged as follows::

                [
                        [V0, d0..dn]
                        [V1, d0..dn]
                        ...
                        [Vm, d0..dn]
                ]

        ``V`` is the absolute coordinate value from the default font, and ``d0-dn``
        are the delta values from the *n* regions. Each ``V`` is an absolute
        coordinate from the default font.

        We want to return a list::

                [
                        [v0, v1..vm]
                        [d0..dn]
                        ...
                        [d0..dn]
                        numBlends
                        blendOp
                ]

        where each ``v`` is relative to the previous default font value.
        """
    numMasters = len(value[0])
    numBlends = len(value)
    numStack = numBlends * numMasters + 1
    if numStack > self.maxBlendStack:
        numBlendValues = int((self.maxBlendStack - 1) / numMasters)
        out = []
        while True:
            numVal = min(len(value), numBlendValues)
            if numVal == 0:
                break
            valList = value[0:numVal]
            out1 = self.arg_delta_blend(valList)
            out.extend(out1)
            value = value[numVal:]
    else:
        firstList = [0] * numBlends
        deltaList = [None] * numBlends
        i = 0
        prevVal = 0
        while i < numBlends:
            defaultValue = value[i][0]
            firstList[i] = defaultValue - prevVal
            prevVal = defaultValue
            deltaList[i] = value[i][1:]
            i += 1
        relValueList = firstList
        for blendList in deltaList:
            relValueList.extend(blendList)
        out = [encodeNumber(val) for val in relValueList]
        out.append(encodeNumber(numBlends))
        out.append(bytechr(blendOp))
    return out