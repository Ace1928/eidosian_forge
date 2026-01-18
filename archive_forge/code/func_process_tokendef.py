from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def process_tokendef(cls, name, tokendefs=None):
    """Preprocess a dictionary of token definitions."""
    processed = cls._all_tokens[name] = {}
    tokendefs = tokendefs or cls.tokens[name]
    for state in list(tokendefs):
        cls._process_state(tokendefs, processed, state)
    return processed