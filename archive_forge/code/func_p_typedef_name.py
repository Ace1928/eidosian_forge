from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_typedef_name(self, p):
    """ typedef_name : TYPEID """
    p[0] = c_ast.IdentifierType([p[1]], coord=self._token_coord(p, 1))