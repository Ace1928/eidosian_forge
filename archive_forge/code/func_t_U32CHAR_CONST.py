import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u32char_const)
def t_U32CHAR_CONST(self, t):
    return t