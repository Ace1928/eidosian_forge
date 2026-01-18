from pyparsing import *
from sys import stdin, argv, exit
def function_end_action(self, text, loc, fun):
    """Code executed at the end of function definition"""
    if DEBUG > 0:
        print('FUN_END:', fun)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.symtab.set_attribute(self.shared.function_index, self.shared.function_params)
    self.symtab.clear_symbols(self.shared.function_index + 1)
    self.codegen.function_end()