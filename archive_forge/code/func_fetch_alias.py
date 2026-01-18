from .error import MarkedYAMLError
from .tokens import *
def fetch_alias(self):
    self.save_possible_simple_key()
    self.allow_simple_key = False
    self.tokens.append(self.scan_anchor(AliasToken))