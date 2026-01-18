from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_labeled_statement_1(self, p):
    """ labeled_statement : ID COLON pragmacomp_or_statement """
    p[0] = c_ast.Label(p[1], p[3], self._token_coord(p, 1))