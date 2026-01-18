from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
def scope_transform(module_node):
    dummy_entry = object()
    for name, type in self.context_types.items():
        old_type_entry = getattr(type, 'entry', dummy_entry)
        entry = module_node.scope.declare_type(name, type, None, visibility='extern')
        if old_type_entry is not dummy_entry:
            type.entry = old_type_entry
        entry.in_cinclude = True
    return module_node