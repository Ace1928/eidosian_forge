from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_postfix_expression_5(self, p):
    """ postfix_expression  : postfix_expression PLUSPLUS
                                | postfix_expression MINUSMINUS
        """
    p[0] = c_ast.UnaryOp('p' + p[2], p[1], p[1].coord)