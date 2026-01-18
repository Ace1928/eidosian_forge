from pyparsing import *
from sys import stdin, argv, exit
def set_type(self, index, stype):
    try:
        self.table[index].type = stype
    except Exception:
        self.error()