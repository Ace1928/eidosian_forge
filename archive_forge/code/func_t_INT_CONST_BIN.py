import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(bin_constant)
def t_INT_CONST_BIN(self, t):
    return t