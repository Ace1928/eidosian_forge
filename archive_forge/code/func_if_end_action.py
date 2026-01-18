from pyparsing import *
from sys import stdin, argv, exit
def if_end_action(self, text, loc, arg):
    """Code executed after recognising a whole if statement"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('IF_END:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.codegen.newline_label('exit{0}'.format(self.label_stack.pop()), True, True)