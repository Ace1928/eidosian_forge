from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def arg_blendList(self, name):
    """
        There may be non-blend args at the top of the stack. We first calculate
        where the blend args start in the stack. These are the last
        numMasters*numBlends) +1 args.
        The blend args starts with numMasters relative coordinate values, the  BlueValues in the list from the default master font. This is followed by
        numBlends list of values. Each of  value in one of these lists is the
        Variable Font delta for the matching region.

        We re-arrange this to be a list of numMaster entries. Each entry starts with the corresponding default font relative value, and is followed by
        the delta values. We then convert the default values, the first item in each entry, to an absolute value.
        """
    vsindex = self.dict.get('vsindex', 0)
    numMasters = self.parent.getNumRegions(vsindex) + 1
    numBlends = self.pop()
    args = self.popall()
    numArgs = len(args)
    assert numArgs == numMasters * numBlends
    value = [None] * numBlends
    numDeltas = numMasters - 1
    i = 0
    prevVal = 0
    while i < numBlends:
        newVal = args[i] + prevVal
        prevVal = newVal
        masterOffset = numBlends + i * numDeltas
        blendList = [newVal] + args[masterOffset:masterOffset + numDeltas]
        value[i] = blendList
        i += 1
    return value