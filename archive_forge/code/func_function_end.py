from pyparsing import *
from sys import stdin, argv, exit
def function_end(self):
    """Inserts an exit label and function return instructions"""
    self.newline_label(self.shared.function_name + '_exit', True, True)
    self.move('%14', '%15')
    self.pop('%14')
    self.newline_text('RET', True)