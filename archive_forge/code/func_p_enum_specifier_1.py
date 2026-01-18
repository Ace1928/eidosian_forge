from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_enum_specifier_1(self, p):
    """ enum_specifier  : ENUM ID
                            | ENUM TYPEID
        """
    p[0] = c_ast.Enum(p[2], None, self._token_coord(p, 1))