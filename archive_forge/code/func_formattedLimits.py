from enum import Enum
from ...Qt import QT_LIB, QtCore
from .list import ListParameter
def formattedLimits(self):
    mapping = self.enumMap
    shortestName = min((len(name) for name in mapping))
    names = list(mapping)
    cmpName, *names = names
    substringEnd = next((ii + 1 for ii in range(-1, -shortestName - 1, -1) if any((cmpName[ii] != curName[ii] for curName in names))), None)
    if substringEnd == 0:
        substringEnd = None
    return {kk[:substringEnd]: vv for kk, vv in self.enumMap.items()}