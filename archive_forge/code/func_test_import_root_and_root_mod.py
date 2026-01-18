import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_root_and_root_mod(self):
    """Test that 'import root, root.mod' can be done.

        The second import should re-use the first one, and just add
        children to be imported.
        """
    try:
        root4
    except NameError:
        pass
    else:
        self.fail('root4 was not supposed to exist yet')
    InstrumentedImportReplacer(scope=globals(), name='root4', module_path=[self.root_name], member=None, children={})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root4, '__class__'))
    children = object.__getattribute__(root4, '_import_replacer_children')
    children['mod4'] = ([self.root_name, self.mod_name], None, {})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root4.mod4, '__class__'))
    self.assertEqual(2, root4.mod4.var2)
    mod_path = self.root_name + '.' + self.mod_name
    self.assertEqual([('__getattribute__', 'mod4'), ('_import', 'root4'), ('import', self.root_name, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod4'), ('import', mod_path, [], 0)], self.actions)