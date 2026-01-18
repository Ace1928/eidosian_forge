import bisect
import configparser
import inspect
import io
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
@register_check
def missing_whitespace_after_keyword(logical_line, tokens):
    """Keywords should be followed by whitespace.

    Okay: from foo import (bar, baz)
    E275: from foo import(bar, baz)
    E275: from importable.module import(bar, baz)
    E275: if(foo): bar
    """
    for tok0, tok1 in zip(tokens, tokens[1:]):
        if tok0.end == tok1.start and tok0.type == tokenize.NAME and keyword.iskeyword(tok0.string) and (tok0.string not in SINGLETONS) and (not (tok0.string == 'except' and tok1.string == '*')) and (not (tok0.string == 'yield' and tok1.string == ')')) and (tok1.string not in ':\n'):
            yield (tok0.end, 'E275 missing whitespace after keyword')