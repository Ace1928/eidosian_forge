from pyparsing import *
from sys import stdin, argv, exit
def symbol(self, index):
    """Generates symbol name from index"""
    if isinstance(index, str):
        return index
    elif index < 0 or index >= self.symtab.table_len:
        self.error('symbol table index out of range')
    sym = self.symtab.table[index]
    if sym.kind == SharedData.KINDS.LOCAL_VAR:
        return '-{0}(1:%14)'.format(sym.attribute * 4 + 4)
    elif sym.kind == SharedData.KINDS.PARAMETER:
        return '{0}(1:%14)'.format(8 + sym.attribute * 4)
    elif sym.kind == SharedData.KINDS.CONSTANT:
        return '${0}'.format(sym.name)
    else:
        return '{0}'.format(sym.name)