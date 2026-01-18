import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def determineexprtype(expr, vars, rules={}):
    if expr in vars:
        return _ensure_exprdict(vars[expr])
    expr = expr.strip()
    if determineexprtype_re_1.match(expr):
        return {'typespec': 'complex'}
    m = determineexprtype_re_2.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess('determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        return {'typespec': 'integer'}
    m = determineexprtype_re_3.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess('determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        return {'typespec': 'real'}
    for op in ['+', '-', '*', '/']:
        for e in [x.strip() for x in markoutercomma(expr, comma=op).split('@' + op + '@')]:
            if e in vars:
                return _ensure_exprdict(vars[e])
    t = {}
    if determineexprtype_re_4.match(expr):
        t = determineexprtype(expr[1:-1], vars, rules)
    else:
        m = determineexprtype_re_5.match(expr)
        if m:
            rn = m.group('name')
            t = determineexprtype(m.group('name'), vars, rules)
            if t and 'attrspec' in t:
                del t['attrspec']
            if not t:
                if rn[0] in rules:
                    return _ensure_exprdict(rules[rn[0]])
    if expr[0] in '\'"':
        return {'typespec': 'character', 'charselector': {'*': '*'}}
    if not t:
        outmess('determineexprtype: could not determine expressions (%s) type.\n' % repr(expr))
    return t