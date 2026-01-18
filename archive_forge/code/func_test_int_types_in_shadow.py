import unittest
from Cython import Shadow
from Cython.Compiler import Options, CythonScope, PyrexTypes, Errors
def test_int_types_in_shadow(self):
    missing_types = []
    for int_name in Shadow.int_types:
        for sign in ['', 'u', 's']:
            name = sign + int_name
            if sign and (int_name in ['Py_UNICODE', 'Py_UCS4', 'Py_ssize_t', 'ssize_t', 'ptrdiff_t', 'Py_hash_t'] or name == 'usize_t'):
                self.assertNotIn(name, dir(Shadow))
                self.assertNotIn('p_' + name, dir(Shadow))
                continue
            if not hasattr(Shadow, name):
                missing_types.append(name)
            for ptr in range(1, 4):
                ptr_name = 'p' * ptr + '_' + name
                if not hasattr(Shadow, ptr_name):
                    missing_types.append(ptr_name)
    self.assertEqual(missing_types, [])