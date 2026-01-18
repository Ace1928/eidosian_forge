from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_labeled_statement_3(self, p):
    """ labeled_statement : DEFAULT COLON pragmacomp_or_statement """
    p[0] = c_ast.Default([p[3]], self._token_coord(p, 1))