from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def p_error(self, p):
    if p.type == 'IDENTIFIER':
        raise self.PythranSpecError('Unexpected identifier `{}` at that point'.format(p.value), p.lexpos)
    else:
        raise self.PythranSpecError('Unexpected token `{}` at that point'.format(p.value), p.lexpos)