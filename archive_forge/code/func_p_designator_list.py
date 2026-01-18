from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_designator_list(self, p):
    """ designator_list : designator
                            | designator_list designator
        """
    p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]