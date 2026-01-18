from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def exprnode_to_known_standard_library_name(node, env):
    qualified_name_parts = []
    known_name = None
    while node.is_attribute:
        qualified_name_parts.append(node.attribute)
        node = node.obj
    if node.is_name:
        entry = env.lookup(node.name)
        if entry and entry.known_standard_library_import:
            if get_known_standard_library_entry(entry.known_standard_library_import):
                known_name = entry.known_standard_library_import
            else:
                standard_env = get_known_standard_library_module_scope(entry.known_standard_library_import)
                if standard_env:
                    qualified_name_parts.append(standard_env.name)
                    known_name = '.'.join(reversed(qualified_name_parts))
    return known_name