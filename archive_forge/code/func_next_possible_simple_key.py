from .error import MarkedYAMLError
from .tokens import *
def next_possible_simple_key(self):
    min_token_number = None
    for level in self.possible_simple_keys:
        key = self.possible_simple_keys[level]
        if min_token_number is None or key.token_number < min_token_number:
            min_token_number = key.token_number
    return min_token_number