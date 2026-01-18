from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_iteration_statement_1(self, p):
    """ iteration_statement : WHILE LPAREN expression RPAREN pragmacomp_or_statement """
    p[0] = c_ast.While(p[3], p[5], self._token_coord(p, 1))