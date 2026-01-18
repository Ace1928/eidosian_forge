import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_lazy_import(self):
    """Smoke test that lazy_import() does the right thing"""
    try:
        root8
    except NameError:
        pass
    else:
        self.fail('root8 was not supposed to exist yet')
    lazy_import.lazy_import(globals(), 'import {} as root8'.format(self.root_name), lazy_import_class=InstrumentedImportReplacer)
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root8, '__class__'))
    self.assertEqual(1, root8.var1)
    self.assertEqual(1, root8.var1)
    self.assertEqual(1, root8.func1(1))
    self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root8'), ('import', self.root_name, [], 0)], self.actions)