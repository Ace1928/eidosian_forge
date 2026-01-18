from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_pppragma_directive_list(self, p):
    """ pppragma_directive_list : pppragma_directive
                                    | pppragma_directive_list pppragma_directive
        """
    p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]