import errno
from .. import osutils, tests
from . import features
def assertReadDir(self, expected, prefix, top_unicode):
    result = self._remove_stat_from_dirblock(self.reader.read_dir(prefix, top_unicode))
    self.assertEqual(expected, result)