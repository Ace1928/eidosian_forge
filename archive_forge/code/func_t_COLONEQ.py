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
def t_COLONEQ(t):
    """:="""
    t.lexer.begin('data')
    return t