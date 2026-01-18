from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_parameter_type_list(self, p):
    """ parameter_type_list : parameter_list
                                | parameter_list COMMA ELLIPSIS
        """
    if len(p) > 2:
        p[1].params.append(c_ast.EllipsisParam(self._token_coord(p, 3)))
    p[0] = p[1]