from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
def mergeBits(bitmap):

    def wrapper(lst):
        lst = list(lst)
        returnValue = 0
        for bitNumber in range(bitmap['size']):
            try:
                mergeLogic = bitmap[bitNumber]
            except KeyError:
                try:
                    mergeLogic = bitmap['*']
                except KeyError:
                    raise Exception("Don't know how to merge bit %s" % bitNumber)
            shiftedBit = 1 << bitNumber
            mergedValue = mergeLogic((bool(item & shiftedBit) for item in lst))
            returnValue |= mergedValue << bitNumber
        return returnValue
    return wrapper