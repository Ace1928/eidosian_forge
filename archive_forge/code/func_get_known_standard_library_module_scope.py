from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def get_known_standard_library_module_scope(module_name):
    mod = _known_module_scopes.get(module_name)
    if mod:
        return mod
    if module_name == 'typing':
        mod = ModuleScope(module_name, None, None)
        for name, tp in [('Dict', dict_type), ('List', list_type), ('Tuple', tuple_type), ('Set', set_type), ('FrozenSet', frozenset_type)]:
            name = EncodedString(name)
            entry = mod.declare_type(name, tp, pos=None)
            var_entry = Entry(name, None, PyrexTypes.py_object_type)
            var_entry.is_pyglobal = True
            var_entry.is_variable = True
            var_entry.scope = mod
            entry.as_variable = var_entry
            entry.known_standard_library_import = '%s.%s' % (module_name, name)
        for name in ['ClassVar', 'Optional']:
            name = EncodedString(name)
            indexed_type = PyrexTypes.SpecialPythonTypeConstructor(EncodedString('typing.' + name))
            entry = mod.declare_type(name, indexed_type, pos=None)
            var_entry = Entry(name, None, PyrexTypes.py_object_type)
            var_entry.is_pyglobal = True
            var_entry.is_variable = True
            var_entry.scope = mod
            entry.as_variable = var_entry
            entry.known_standard_library_import = '%s.%s' % (module_name, name)
        _known_module_scopes[module_name] = mod
    elif module_name == 'dataclasses':
        mod = ModuleScope(module_name, None, None)
        indexed_type = PyrexTypes.SpecialPythonTypeConstructor(EncodedString('dataclasses.InitVar'))
        initvar_string = EncodedString('InitVar')
        entry = mod.declare_type(initvar_string, indexed_type, pos=None)
        var_entry = Entry(initvar_string, None, PyrexTypes.py_object_type)
        var_entry.is_pyglobal = True
        var_entry.scope = mod
        entry.as_variable = var_entry
        entry.known_standard_library_import = '%s.InitVar' % module_name
        for name in ['dataclass', 'field']:
            mod.declare_var(EncodedString(name), PyrexTypes.py_object_type, pos=None)
        _known_module_scopes[module_name] = mod
    elif module_name == 'functools':
        mod = ModuleScope(module_name, None, None)
        for name in ['total_ordering']:
            mod.declare_var(EncodedString(name), PyrexTypes.py_object_type, pos=None)
        _known_module_scopes[module_name] = mod
    return mod