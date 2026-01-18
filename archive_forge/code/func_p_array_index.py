from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_array_index(self, p):
    if len(p) == 3:
        p[0] = slice(0, -1, -1)
    elif len(p) == 1 or p[1] == ':':
        p[0] = slice(0, -1, 1)
    else:
        p[0] = slice(0, int(p[1]), 1)