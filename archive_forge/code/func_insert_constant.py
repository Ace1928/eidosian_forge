from pyparsing import *
from sys import stdin, argv, exit
def insert_constant(self, cname, ctype):
    """Inserts a constant (or returns index if the constant already exists)
           Additionally, checks for range.
        """
    index = self.lookup_symbol(cname, stype=ctype)
    if index == None:
        num = int(cname)
        if ctype == SharedData.TYPES.INT:
            if num < SharedData.MIN_INT or num > SharedData.MAX_INT:
                raise SemanticException("Integer constant '%s' out of range" % cname)
        elif ctype == SharedData.TYPES.UNSIGNED:
            if num < 0 or num > SharedData.MAX_UNSIGNED:
                raise SemanticException("Unsigned constant '%s' out of range" % cname)
        index = self.insert_symbol(cname, SharedData.KINDS.CONSTANT, ctype)
    return index