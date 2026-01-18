from pyparsing import *
from sys import stdin, argv, exit
def local_variable_action(self, text, loc, var):
    """Code executed after recognising a local variable"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('LOCAL_VAR:', var, var.name, var.type)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    index = self.symtab.insert_local_var(var.name, var.type, self.shared.function_vars)
    self.shared.function_vars += 1
    return index