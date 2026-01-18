from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_or_union_specifier_3(self, p):
    """ struct_or_union_specifier   : struct_or_union ID brace_open struct_declaration_list brace_close
                                        | struct_or_union ID brace_open brace_close
                                        | struct_or_union TYPEID brace_open struct_declaration_list brace_close
                                        | struct_or_union TYPEID brace_open brace_close
        """
    klass = self._select_struct_union_class(p[1])
    if len(p) == 5:
        p[0] = klass(name=p[2], decls=[], coord=self._token_coord(p, 2))
    else:
        p[0] = klass(name=p[2], decls=p[4], coord=self._token_coord(p, 2))