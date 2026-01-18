from pyparsing import *
from sys import stdin, argv, exit
def function_body(self):
    """Inserts a local variable initialization and body label"""
    if self.shared.function_vars > 0:
        const = self.symtab.insert_constant('0{}'.format(self.shared.function_vars * 4), SharedData.TYPES.UNSIGNED)
        self.arithmetic('-', '%15', const, '%15')
    self.newline_label(self.shared.function_name + '_body', True, True)