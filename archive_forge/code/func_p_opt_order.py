from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_opt_order(self, p):
    if len(p) > 1:
        if p[3] not in 'CF':
            msg = "Invalid Pythran spec. Unknown order '{}'".format(p[3])
            self.p_error(p, msg)
        p[0] = p[3]
    else:
        p[0] = None