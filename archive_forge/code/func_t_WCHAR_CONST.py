import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(wchar_const)
def t_WCHAR_CONST(self, t):
    return t