from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def missingRanges(charList):
    rv = []
    if charList[0] != 0:
        rv.append([0, charList[0][0] - 1])
    for i, item in enumerate(charList[:-1]):
        rv.append([item[1] + 1, charList[i + 1][0] - 1])
    if charList[-1][1] != max_unicode:
        rv.append([charList[-1][1] + 1, max_unicode])
    return rv