from pyparsing import *
from sys import stdin, argv, exit
def same_types(self, index1, index2):
    """Returns True if both symbol table elements are of the same type"""
    try:
        same = self.table[index1].type == self.table[index2].type != SharedData.TYPES.NO_TYPE
    except Exception:
        self.error()
    return same