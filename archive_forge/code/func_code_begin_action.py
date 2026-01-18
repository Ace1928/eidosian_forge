from pyparsing import *
from sys import stdin, argv, exit
def code_begin_action(self):
    """Inserts text at start of code segment"""
    self.codegen.prepare_code_segment()