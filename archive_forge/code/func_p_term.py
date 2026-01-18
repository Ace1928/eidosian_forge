from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_term(self, p):
    if p[1] == 'str':
        p[0] = str
    elif p[1] == 'slice':
        p[0] = slice
    elif p[1] == 'None':
        p[0] = type(None)
    else:
        p[0] = p[1][0]