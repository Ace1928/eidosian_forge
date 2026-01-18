from .error import MarkedYAMLError
from .tokens import *
def fetch_directive(self):
    self.unwind_indent(-1)
    self.remove_possible_simple_key()
    self.allow_simple_key = False
    self.tokens.append(self.scan_directive())