from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_initializer_2(self, p):
    """ initializer : brace_open initializer_list_opt brace_close
                        | brace_open initializer_list COMMA brace_close
        """
    if p[2] is None:
        p[0] = c_ast.InitList([], self._token_coord(p, 1))
    else:
        p[0] = p[2]