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
def p_expr(p):
    """expr : statements
    |"""
    if len(p) == 2:
        for stmt in p[1]:
            if type(stmt) is list:
                _parse_info[None].append(stmt)
            else:
                for key in stmt:
                    if key in _parse_info:
                        _parse_info[key].append(stmt[key])
                    else:
                        _parse_info[key] = stmt[key]