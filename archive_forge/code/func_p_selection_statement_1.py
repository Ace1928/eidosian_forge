from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_selection_statement_1(self, p):
    """ selection_statement : IF LPAREN expression RPAREN pragmacomp_or_statement """
    p[0] = c_ast.If(p[3], p[5], None, self._token_coord(p, 1))