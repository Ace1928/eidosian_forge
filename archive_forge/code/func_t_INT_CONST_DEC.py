import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(decimal_constant)
def t_INT_CONST_DEC(self, t):
    return t