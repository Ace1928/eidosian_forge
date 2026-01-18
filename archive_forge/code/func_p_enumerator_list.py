from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_enumerator_list(self, p):
    """ enumerator_list : enumerator
                            | enumerator_list COMMA
                            | enumerator_list COMMA enumerator
        """
    if len(p) == 2:
        p[0] = c_ast.EnumeratorList([p[1]], p[1].coord)
    elif len(p) == 3:
        p[0] = p[1]
    else:
        p[1].enumerators.append(p[3])
        p[0] = p[1]