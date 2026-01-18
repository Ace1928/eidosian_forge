import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def get_css_prefix(self, arg):
    if arg is None:
        arg = 'cssclass' in self.options and '.' + self.cssclass or ''
    if isinstance(arg, str):
        args = [arg]
    else:
        args = list(arg)

    def prefix(cls):
        if cls:
            cls = '.' + cls
        tmp = []
        for arg in args:
            tmp.append((arg and arg + ' ' or '') + cls)
        return ', '.join(tmp)
    return prefix