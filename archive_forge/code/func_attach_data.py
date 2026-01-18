from pythran.passmanager import Transformation
import pythran.metadata as metadata
from pythran.spec import parse_pytypes
from pythran.types.conversion import pytype_to_ctype
from pythran.utils import isstr
from gast import AST
import gast as ast
import re
def attach_data(self, node):
    """Generic method called for visit_XXXX() with XXXX in
        GatherOMPData.statements list

        """
    if self.current:
        for curr in self.current:
            md = OMPDirective(curr)
            metadata.add(node, md)
        self.current = list()
    for field_name, field in ast.iter_fields(node):
        if field_name in GatherOMPData.statement_lists:
            if field and isinstance(field[-1], ast.Expr) and self.isompdirective(field[-1].value):
                field.append(ast.Pass())
    self.generic_visit(node)
    directives = metadata.get(node, OMPDirective)
    field_names = {n for n, _ in ast.iter_fields(node)}
    has_no_scope = field_names.isdisjoint(GatherOMPData.statement_lists)
    if directives and has_no_scope:
        sdirective = ''.join((d.s for d in directives))
        scoping = ('parallel', 'task', 'section')
        if any((s in sdirective for s in scoping)):
            metadata.clear(node, OMPDirective)
            node = ast.If(ast.Constant(1, None), [node], [])
            for directive in directives:
                metadata.add(node, directive)
    return node