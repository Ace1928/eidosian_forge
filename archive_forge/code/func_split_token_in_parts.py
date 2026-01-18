from __future__ import unicode_literals
from .base import DEFAULT_ATTRS, Attrs
def split_token_in_parts(token):
    """
    Take a Token, and turn it in a list of tokens, by splitting
    it on ':' (taking that as a separator.)
    """
    result = []
    current = []
    for part in token + (':',):
        if part == ':':
            if current:
                result.append(tuple(current))
                current = []
        else:
            current.append(part)
    return result