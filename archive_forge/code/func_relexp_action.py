from pyparsing import *
from sys import stdin, argv, exit
def relexp_action(self, text, loc, arg):
    """Code executed after recognising a relexp expression (something relop something)"""
    if DEBUG > 0:
        print('REL_EXP:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    exshared.setpos(loc, text)
    if not self.symtab.same_types(arg[0], arg[2]):
        raise SemanticException("Invalid operands for operator '{0}'".format(arg[1]))
    self.codegen.compare(arg[0], arg[2])
    self.relexp_code = self.codegen.relop_code(arg[1], self.symtab.get_type(arg[0]))
    return self.relexp_code