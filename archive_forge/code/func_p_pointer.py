from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_pointer(self, p):
    """ pointer : TIMES type_qualifier_list_opt
                    | TIMES type_qualifier_list_opt pointer
        """
    coord = self._token_coord(p, 1)
    nested_type = c_ast.PtrDecl(quals=p[2] or [], type=None, coord=coord)
    if len(p) > 3:
        tail_type = p[3]
        while tail_type.type is not None:
            tail_type = tail_type.type
        tail_type.type = nested_type
        p[0] = p[3]
    else:
        p[0] = nested_type