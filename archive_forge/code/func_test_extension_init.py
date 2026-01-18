import unittest
import os
import warnings
from test.support.warnings_helper import check_warnings
from distutils.extension import read_setup_file, Extension
def test_extension_init(self):
    self.assertRaises(AssertionError, Extension, 1, [])
    ext = Extension('name', [])
    self.assertEqual(ext.name, 'name')
    self.assertRaises(AssertionError, Extension, 'name', 'file')
    self.assertRaises(AssertionError, Extension, 'name', ['file', 1])
    ext = Extension('name', ['file1', 'file2'])
    self.assertEqual(ext.sources, ['file1', 'file2'])
    for attr in ('include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends'):
        self.assertEqual(getattr(ext, attr), [])
    self.assertEqual(ext.language, None)
    self.assertEqual(ext.optional, None)
    with check_warnings() as w:
        warnings.simplefilter('always')
        ext = Extension('name', ['file1', 'file2'], chic=True)
    self.assertEqual(len(w.warnings), 1)
    self.assertEqual(str(w.warnings[0].message), "Unknown Extension options: 'chic'")