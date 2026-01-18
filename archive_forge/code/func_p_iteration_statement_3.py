from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_iteration_statement_3(self, p):
    """ iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement """
    p[0] = c_ast.For(p[3], p[5], p[7], p[9], self._token_coord(p, 1))