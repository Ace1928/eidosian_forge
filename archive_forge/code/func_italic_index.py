from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def italic_index(c):
    return data[char_info(c) + 2] // 4