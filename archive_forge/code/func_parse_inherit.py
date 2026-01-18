from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_inherit(tokens, name, context):
    first, pos = tokens[0]
    assert first.startswith('inherit ')
    expr = first.split(None, 1)[1]
    return (('inherit', pos, expr), tokens[1:])