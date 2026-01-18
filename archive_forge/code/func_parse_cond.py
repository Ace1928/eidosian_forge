from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_cond(tokens, name, context):
    start = tokens[0][1]
    pieces = []
    context = context + ('if',)
    while 1:
        if not tokens:
            raise TemplateError('Missing {{endif}}', position=start, name=name)
        if isinstance(tokens[0], tuple) and tokens[0][0] == 'endif':
            return (('cond', start) + tuple(pieces), tokens[1:])
        next_chunk, tokens = parse_one_cond(tokens, name, context)
        pieces.append(next_chunk)