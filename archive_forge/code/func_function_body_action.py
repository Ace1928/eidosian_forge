from pyparsing import *
from sys import stdin, argv, exit
def function_body_action(self, text, loc, fun):
    """Code executed after recognising the beginning of function's body"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('FUN_BODY:', fun)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.codegen.function_body()