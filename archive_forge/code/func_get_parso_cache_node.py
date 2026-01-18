import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_parso_cache_node(grammar, path):
    """
    This is of course not public. But as long as I control parso, this
    shouldn't be a problem. ~ Dave

    The reason for this is mostly caching. This is obviously also a sign of a
    broken caching architecture.
    """
    return parser_cache[grammar._hashed][path]