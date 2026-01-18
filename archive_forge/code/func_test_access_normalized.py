import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test_access_normalized(self):
    files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3', squared_c + '.4', quarter_c + '.5']
    try:
        self.build_tree(files, line_endings='native')
    except UnicodeError:
        raise TestSkipped('filesystem cannot create unicode files')
    for fname in files:
        path, can_access = osutils.normalized_filename(fname)
        self.assertEqual(path, fname)
        self.assertTrue(can_access)
        with open(path, 'rb') as f:
            shouldbe = b'contents of %s%s' % (path.encode('utf8'), os.linesep.encode('utf-8'))
            actual = f.read()
        self.assertEqual(shouldbe, actual, 'contents of %r is incorrect: %r != %r' % (path, shouldbe, actual))