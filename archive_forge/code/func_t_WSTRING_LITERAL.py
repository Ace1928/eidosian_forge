import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(wstring_literal)
def t_WSTRING_LITERAL(self, t):
    return t