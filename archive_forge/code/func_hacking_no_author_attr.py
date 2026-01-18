import re
import tokenize
from hacking import core
import pycodestyle
@core.flake8ext
def hacking_no_author_attr(logical_line, tokens):
    """__author__ should not be used.

    S362: __author__ = slukjanov
    """
    for token_type, text, start_index, _, _ in tokens:
        if token_type == tokenize.NAME and text == '__author__':
            yield (start_index[1], 'S362: __author__ should not be used')