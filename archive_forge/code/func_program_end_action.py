from pyparsing import *
from sys import stdin, argv, exit
def program_end_action(self, text, loc, arg):
    """Checks if there is a 'main' function and the type of 'main' function"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('PROGRAM_END:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    index = self.symtab.lookup_symbol('main', SharedData.KINDS.FUNCTION)
    if index == None:
        raise SemanticException("Undefined reference to 'main'", False)
    elif self.symtab.get_type(index) != SharedData.TYPES.INT:
        self.warning("Return type of 'main' is not int", False)