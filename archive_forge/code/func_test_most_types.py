import unittest
from Cython import Shadow
from Cython.Compiler import Options, CythonScope, PyrexTypes, Errors
def test_most_types(self):
    cython_scope = CythonScope.create_cython_scope(None)

    class Context:
        cpp = False
        language_level = 3
        future_directives = []
    cython_scope.context = Context
    Errors.init_thread()
    missing_types = []
    missing_lookups = []
    for (signed, longness, name), type_ in PyrexTypes.modifiers_and_name_to_type.items():
        if name == 'object':
            continue
        if not hasattr(Shadow, name):
            missing_types.append(name)
        if not cython_scope.lookup_type(name):
            missing_lookups.append(name)
        for ptr in range(1, 4):
            ptr_name = 'p' * ptr + '_' + name
            if not hasattr(Shadow, ptr_name):
                missing_types.append(ptr_name)
            if not cython_scope.lookup_type(ptr_name):
                missing_lookups.append(ptr_name)
    self.assertEqual(missing_types, [])
    self.assertEqual(missing_lookups, [])