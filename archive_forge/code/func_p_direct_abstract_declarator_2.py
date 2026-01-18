from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_direct_abstract_declarator_2(self, p):
    """ direct_abstract_declarator  : direct_abstract_declarator LBRACKET assignment_expression_opt RBRACKET
        """
    arr = c_ast.ArrayDecl(type=None, dim=p[3], dim_quals=[], coord=p[1].coord)
    p[0] = self._type_modify_decl(decl=p[1], modifier=arr)