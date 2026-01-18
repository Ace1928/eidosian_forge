import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def create_modules(self):
    """Create some random modules to be imported.

        Each entry has a random suffix, and the full names are saved

        These are setup as follows:
         base/ <= used to ensure not in default search path
            root-XXX/
                __init__.py <= This will contain var1, func1
                mod-XXX.py <= This will contain var2, func2
                sub-XXX/
                    __init__.py <= Contains var3, func3
                    submoda-XXX.py <= contains var4, func4
                    submodb-XXX.py <= containse var5, func5
        """
    rand_suffix = osutils.rand_chars(4)
    root_name = 'root_' + rand_suffix
    mod_name = 'mod_' + rand_suffix
    sub_name = 'sub_' + rand_suffix
    submoda_name = 'submoda_' + rand_suffix
    submodb_name = 'submodb_' + rand_suffix
    os.mkdir('base')
    root_path = osutils.pathjoin('base', root_name)
    os.mkdir(root_path)
    root_init = osutils.pathjoin(root_path, '__init__.py')
    with open(osutils.pathjoin(root_path, '__init__.py'), 'w') as f:
        f.write('var1 = 1\ndef func1(a):\n  return a\n')
    mod_path = osutils.pathjoin(root_path, mod_name + '.py')
    with open(mod_path, 'w') as f:
        f.write('var2 = 2\ndef func2(a):\n  return a\n')
    sub_path = osutils.pathjoin(root_path, sub_name)
    os.mkdir(sub_path)
    with open(osutils.pathjoin(sub_path, '__init__.py'), 'w') as f:
        f.write('var3 = 3\ndef func3(a):\n  return a\n')
    submoda_path = osutils.pathjoin(sub_path, submoda_name + '.py')
    with open(submoda_path, 'w') as f:
        f.write('var4 = 4\ndef func4(a):\n  return a\n')
    submodb_path = osutils.pathjoin(sub_path, submodb_name + '.py')
    with open(submodb_path, 'w') as f:
        f.write('var5 = 5\ndef func5(a):\n  return a\n')
    self.root_name = root_name
    self.mod_name = mod_name
    self.sub_name = sub_name
    self.submoda_name = submoda_name
    self.submodb_name = submodb_name