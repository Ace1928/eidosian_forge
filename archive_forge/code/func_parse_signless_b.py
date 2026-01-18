import re
from .parser import _next_significant, _to_token_iterator
def parse_signless_b(tokens, a, b_sign):
    token = _next_significant(tokens)
    if token.type == 'number' and token.is_integer and (not token.representation[0] in '-+'):
        return parse_end(tokens, a, b_sign * token.int_value)