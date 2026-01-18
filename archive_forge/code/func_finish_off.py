import re
from pgen2 import grammar, token
def finish_off(self):
    """Create additional useful structures.  (Internal)."""
    self.keywords = {}
    self.tokens = {}
    for ilabel, (type, value) in enumerate(self.labels):
        if type == token.NAME and value is not None:
            self.keywords[value] = ilabel
        elif value is None:
            self.tokens[type] = ilabel