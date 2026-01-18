from .error import MarkedYAMLError
from .tokens import *
def check_key(self):
    if self.flow_level:
        return True
    else:
        return self.peek(1) in '\x00 \t\r\n\x85\u2028\u2029'