from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def declare_tuple_type(self, pos, components):
    components = tuple(components)
    try:
        ttype = self._cached_tuple_types[components]
    except KeyError:
        ttype = self._cached_tuple_types[components] = PyrexTypes.c_tuple_type(components)
    cname = ttype.cname
    entry = self.lookup_here(cname)
    if not entry:
        scope = StructOrUnionScope(cname)
        for ix, component in enumerate(components):
            scope.declare_var(name='f%s' % ix, type=component, pos=pos)
        struct_entry = self.declare_struct_or_union(cname + '_struct', 'struct', scope, typedef_flag=True, pos=pos, cname=cname)
        self.type_entries.remove(struct_entry)
        ttype.struct_entry = struct_entry
        entry = self.declare_type(cname, ttype, pos, cname)
    ttype.entry = entry
    return entry