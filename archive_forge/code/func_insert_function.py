from pyparsing import *
from sys import stdin, argv, exit
def insert_function(self, fname, ftype):
    """Inserts a new function"""
    index = self.insert_id(fname, SharedData.KINDS.FUNCTION, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.FUNCTION], ftype)
    self.table[index].set_attribute('Params', 0)
    return index