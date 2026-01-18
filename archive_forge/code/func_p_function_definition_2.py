from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_function_definition_2(self, p):
    """ function_definition : declaration_specifiers id_declarator declaration_list_opt compound_statement
        """
    spec = p[1]
    p[0] = self._build_function_definition(spec=spec, decl=p[2], param_decls=p[3], body=p[4])