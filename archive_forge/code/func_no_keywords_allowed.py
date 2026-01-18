import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
def no_keywords_allowed(s, l, t):
    wd = t[0]
    return not keyword.iskeyword(wd)