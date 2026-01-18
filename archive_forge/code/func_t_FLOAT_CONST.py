import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(floating_constant)
def t_FLOAT_CONST(self, t):
    return t