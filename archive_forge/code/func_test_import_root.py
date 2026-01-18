import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_root(self):
    """Test 'import root-XXX as root1'"""
    try:
        root1
    except NameError:
        pass
    else:
        self.fail('root1 was not supposed to exist yet')
    InstrumentedImportReplacer(scope=globals(), name='root1', module_path=[self.root_name], member=None, children={})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root1, '__class__'))
    self.assertEqual(1, root1.var1)
    self.assertEqual('x', root1.func1('x'))
    self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root1'), ('import', self.root_name, [], 0)], self.actions)