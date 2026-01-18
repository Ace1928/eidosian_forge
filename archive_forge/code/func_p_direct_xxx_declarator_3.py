from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
@parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
def p_direct_xxx_declarator_3(self, p):
    """ direct_xxx_declarator   : direct_xxx_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET
        """
    quals = (p[3] if len(p) > 5 else []) or []
    arr = c_ast.ArrayDecl(type=None, dim=p[4] if len(p) > 5 else p[3], dim_quals=quals, coord=p[1].coord)
    p[0] = self._type_modify_decl(decl=p[1], modifier=arr)