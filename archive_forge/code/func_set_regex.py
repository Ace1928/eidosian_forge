import re
import sys
import types
import copy
import os
import inspect
def set_regex(f):
    if hasattr(r, '__call__'):
        f.regex = _get_regex(r)
    else:
        f.regex = r
    return f