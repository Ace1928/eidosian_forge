from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_expression_statement(self, p):
    """ expression_statement : expression_opt SEMI """
    if p[1] is None:
        p[0] = c_ast.EmptyStatement(self._token_coord(p, 2))
    else:
        p[0] = p[1]