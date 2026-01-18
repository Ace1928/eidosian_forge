from pyparsing import *
from sys import stdin, argv, exit
def lookup_id_action(self, text, loc, var):
    """Code executed after recognising an identificator in expression"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('EXP_VAR:', var)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    var_index = self.symtab.lookup_symbol(var.name, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.PARAMETER, SharedData.KINDS.LOCAL_VAR])
    if var_index == None:
        raise SemanticException("'%s' undefined" % var.name)
    return var_index