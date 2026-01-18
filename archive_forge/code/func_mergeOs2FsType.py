from fontTools import ttLib, cffLib
from fontTools.misc.psCharStrings import T2WidthExtractor
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.cmap import computeMegaCmap
from fontTools.merge.util import *
import logging
def mergeOs2FsType(lst):
    lst = list(lst)
    if all((item == 0 for item in lst)):
        return 0
    for i in range(len(lst)):
        if lst[i] & 12:
            lst[i] &= ~2
        elif lst[i] & 8:
            lst[i] |= 4
        elif lst[i] == 0:
            lst[i] = 12
    fsType = mergeBits(os2FsTypeMergeBitMap)(lst)
    if fsType & 2:
        fsType &= ~12
    return fsType