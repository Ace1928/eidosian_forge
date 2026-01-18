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
def p_statements(p):
    """statements : statements statement
    | statement
    | statements NAMESPACE WORD LBRACE statements RBRACE
    | NAMESPACE WORD LBRACE statements RBRACE"""
    len_p = len(p)
    if len_p == 3:
        p[0] = p[1]
        if p[2] is not None:
            p[0].append(p[2])
    elif len_p == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]
    elif len_p == 7:
        p[0] = p[1]
        p[0].append({p[3]: p[5]})
    else:
        p[0] = [{p[2]: p[4]}]