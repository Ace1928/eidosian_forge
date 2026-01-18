from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_enum_specifier_2(self, p):
    """ enum_specifier  : ENUM brace_open enumerator_list brace_close
        """
    p[0] = c_ast.Enum(None, p[3], self._token_coord(p, 1))