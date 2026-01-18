import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(decimal_constant)
def t_ppline_LINE_NUMBER(self, t):
    if self.pp_line is None:
        self.pp_line = t.value
    else:
        pass