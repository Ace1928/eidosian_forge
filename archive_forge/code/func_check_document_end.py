from .error import MarkedYAMLError
from .tokens import *
def check_document_end(self):
    if self.column == 0:
        if self.prefix(3) == '...' and self.peek(3) in '\x00 \t\r\n\x85\u2028\u2029':
            return True