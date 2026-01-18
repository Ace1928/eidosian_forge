from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_function_definition_1(self, p):
    """ function_definition : id_declarator declaration_list_opt compound_statement
        """
    spec = dict(qual=[], alignment=[], storage=[], type=[c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))], function=[])
    p[0] = self._build_function_definition(spec=spec, decl=p[1], param_decls=p[2], body=p[3])