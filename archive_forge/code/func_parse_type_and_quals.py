from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def parse_type_and_quals(self, cdecl):
    ast, macros = self._parse('void __dummy(\n%s\n);' % cdecl)[:2]
    assert not macros
    exprnode = ast.ext[-1].type.args.params[0]
    if isinstance(exprnode, pycparser.c_ast.ID):
        raise CDefError("unknown identifier '%s'" % (exprnode.name,))
    return self._get_type_and_quals(exprnode.type)