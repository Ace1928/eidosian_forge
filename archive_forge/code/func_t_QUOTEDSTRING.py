import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple
@lex.TOKEN('|'.join([_re_quoted_str, _re_quoted_str.replace('"', "'")]))
def t_QUOTEDSTRING(t):
    t.value = '"' + t.value[1:-1].replace(2 * t.value[0], t.value[0]) + '"'
    return t