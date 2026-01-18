from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def exten(c):
    return 4 * (exten_base + remainder(c))