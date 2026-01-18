from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_array_type(self, p):
    if len(p) == 2:
        p[0] = (p[1][0],)
    elif len(p) == 5 and p[4] == ']':

        def args(t):
            return t.__args__ if isinstance(t, NDArray) else (t,)
        p[0] = tuple((NDArray[args(t) + p[3]] for t in p[1]))