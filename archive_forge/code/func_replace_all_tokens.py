from __future__ import unicode_literals
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from collections import defaultdict, namedtuple
def replace_all_tokens(self, token):
    """
        For all the characters in the screen. Set the token to the given `token`.
        """
    b = self.data_buffer
    for y, row in b.items():
        for x, char in row.items():
            b[y][x] = _CHAR_CACHE[char.char, token]