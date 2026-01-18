from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_direct_abstract_declarator_3(self, p):
    """ direct_abstract_declarator  : LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET
        """
    quals = (p[2] if len(p) > 4 else []) or []
    p[0] = c_ast.ArrayDecl(type=c_ast.TypeDecl(None, None, None, None), dim=p[3] if len(p) > 4 else p[2], dim_quals=quals, coord=self._token_coord(p, 1))