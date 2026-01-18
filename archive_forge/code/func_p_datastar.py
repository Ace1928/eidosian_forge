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
def p_datastar(p):
    """
    datastar : data
             |
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = []