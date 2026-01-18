import re
from .parser import _next_significant, _to_token_iterator
def parse_end(tokens, a, b):
    if _next_significant(tokens) is None:
        return (a, b)