from numba.tests.support import TestCase, unittest
from numba.misc import findlib
def test_find_file_nonexistent_path(self):
    candidates = findlib.find_file('libirrelevant.so', 'NONEXISTENT')
    self.assertEqual(candidates, [])