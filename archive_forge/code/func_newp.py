import ctypes, ctypes.util, operator, sys
from . import model
def newp(self, BType, source):
    if not issubclass(BType, CTypesData):
        raise TypeError
    return BType._newp(source)