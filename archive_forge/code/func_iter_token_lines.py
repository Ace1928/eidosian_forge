from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def iter_token_lines(tokenlist):
    """
    Iterator that yields tokenlists for each line.
    """
    line = []
    for token, c in explode_tokens(tokenlist):
        line.append((token, c))
        if c == '\n':
            yield line
            line = []
    yield line