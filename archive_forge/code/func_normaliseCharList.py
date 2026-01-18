from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def normaliseCharList(charList):
    charList = sorted(charList)
    for item in charList:
        assert item[1] >= item[0]
    rv = []
    i = 0
    while i < len(charList):
        j = 1
        rv.append(charList[i])
        while i + j < len(charList) and charList[i + j][0] <= rv[-1][1] + 1:
            rv[-1][1] = charList[i + j][1]
            j += 1
        i += j
    return rv