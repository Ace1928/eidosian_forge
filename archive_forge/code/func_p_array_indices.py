from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_array_indices(self, p):
    if len(p) == 2:
        p[0] = (p[1],)
    else:
        p[0] = (p[1],) + p[3]