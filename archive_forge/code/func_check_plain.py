from .error import MarkedYAMLError
from .tokens import *
def check_plain(self):
    ch = self.peek()
    return ch not in '\x00 \t\r\n\x85\u2028\u2029-?:,[]{}#&*!|>\'"%@`' or (self.peek(1) not in '\x00 \t\r\n\x85\u2028\u2029' and (ch == '-' or (not self.flow_level and ch in '?:')))