from .error import MarkedYAMLError
from .tokens import *
def fetch_block_scalar(self, style):
    self.allow_simple_key = True
    self.remove_possible_simple_key()
    self.tokens.append(self.scan_block_scalar(style))