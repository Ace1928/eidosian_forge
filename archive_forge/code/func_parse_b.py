import re
from .parser import _next_significant, _to_token_iterator
def parse_b(tokens, a):
    token = _next_significant(tokens)
    if token is None:
        return (a, 0)
    elif token == '+':
        return parse_signless_b(tokens, a, 1)
    elif token == '-':
        return parse_signless_b(tokens, a, -1)
    elif token.type == 'number' and token.is_integer and (token.representation[0] in '-+'):
        return parse_end(tokens, a, token.int_value)