import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test_access_non_normalized(self):
    files = [a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3']
    try:
        self.build_tree(files)
    except UnicodeError:
        raise TestSkipped('filesystem cannot create unicode files')
    for fname in files:
        path, can_access = osutils.normalized_filename(fname)
        self.assertNotEqual(path, fname)
        f = open(fname, 'rb')
        f.close()
        if can_access:
            f = open(path, 'rb')
            f.close()
        else:
            self.assertRaises(IOError, open, path, 'rb')