import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u8char_const)
def t_U8CHAR_CONST(self, t):
    return t