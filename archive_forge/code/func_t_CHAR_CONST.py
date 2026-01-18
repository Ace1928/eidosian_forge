import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(char_const)
def t_CHAR_CONST(self, t):
    return t