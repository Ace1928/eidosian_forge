from pyparsing import *
from sys import stdin, argv, exit
def insert_global_var(self, vname, vtype):
    """Inserts a new global variable"""
    return self.insert_id(vname, SharedData.KINDS.GLOBAL_VAR, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.FUNCTION], vtype)