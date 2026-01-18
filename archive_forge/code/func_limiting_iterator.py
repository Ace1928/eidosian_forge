import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def limiting_iterator():
    for i, t in enumerate(iterable):
        if 0 <= max_count <= i:
            raise exceptions.CollectionTooLargeException(max_count)
        yield t