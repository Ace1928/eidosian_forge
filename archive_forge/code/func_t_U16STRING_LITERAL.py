import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u16string_literal)
def t_U16STRING_LITERAL(self, t):
    return t