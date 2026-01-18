import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_mod_from_root(self):
    """Test 'from root-XXX import mod-XXX as mod2'"""
    try:
        mod2
    except NameError:
        pass
    else:
        self.fail('mod2 was not supposed to exist yet')
    InstrumentedImportReplacer(scope=globals(), name='mod2', module_path=[self.root_name], member=self.mod_name, children={})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(mod2, '__class__'))
    self.assertEqual(2, mod2.var2)
    self.assertEqual('y', mod2.func2('y'))
    self.assertEqual([('__getattribute__', 'var2'), ('_import', 'mod2'), ('import', self.root_name, [self.mod_name], 0)], self.actions)