from .error import MarkedYAMLError
from .tokens import *
def remove_possible_simple_key(self):
    if self.flow_level in self.possible_simple_keys:
        key = self.possible_simple_keys[self.flow_level]
        if key.required:
            raise ScannerError('while scanning a simple key', key.mark, "could not find expected ':'", self.get_mark())
        del self.possible_simple_keys[self.flow_level]