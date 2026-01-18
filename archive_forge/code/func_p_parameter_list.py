from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_parameter_list(self, p):
    """ parameter_list  : parameter_declaration
                            | parameter_list COMMA parameter_declaration
        """
    if len(p) == 2:
        p[0] = c_ast.ParamList([p[1]], p[1].coord)
    else:
        p[1].params.append(p[3])
        p[0] = p[1]