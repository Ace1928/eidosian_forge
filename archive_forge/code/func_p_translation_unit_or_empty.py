from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_translation_unit_or_empty(self, p):
    """ translation_unit_or_empty   : translation_unit
                                        | empty
        """
    if p[1] is None:
        p[0] = c_ast.FileAST([])
    else:
        p[0] = c_ast.FileAST(p[1])