from pyparsing import *
from sys import stdin, argv, exit
def assignment_action(self, text, loc, assign):
    """Code executed after recognising an assignment statement"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('ASSIGN:', assign)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    var_index = self.symtab.lookup_symbol(assign.var, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.PARAMETER, SharedData.KINDS.LOCAL_VAR])
    if var_index == None:
        raise SemanticException("Undefined lvalue '%s' in assignment" % assign.var)
    if not self.symtab.same_types(var_index, assign.exp[0]):
        raise SemanticException('Incompatible types in assignment')
    self.codegen.move(assign.exp[0], var_index)