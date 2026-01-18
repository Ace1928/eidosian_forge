from pyparsing import *
from sys import stdin, argv, exit
def function_begin(self):
    """Inserts function name label and function frame initialization"""
    self.newline_label(self.shared.function_name, False, True)
    self.push('%14')
    self.move('%15', '%14')