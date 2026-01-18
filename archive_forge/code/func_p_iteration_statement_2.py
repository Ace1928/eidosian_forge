from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_iteration_statement_2(self, p):
    """ iteration_statement : DO pragmacomp_or_statement WHILE LPAREN expression RPAREN SEMI """
    p[0] = c_ast.DoWhile(p[5], p[2], self._token_coord(p, 1))