import errno
from .. import osutils, tests
from . import features
def assertWalkdirs(self, expected, top, prefix=''):
    old_selected_dir_reader = osutils._selected_dir_reader
    try:
        osutils._selected_dir_reader = self.reader
        finder = osutils._walkdirs_utf8(top, prefix=prefix)
        result = []
        for dirname, dirblock in finder:
            dirblock = self._remove_stat_from_dirblock(dirblock)
            result.append((dirname, dirblock))
        self.assertEqual(expected, result)
    finally:
        osutils._selected_dir_reader = old_selected_dir_reader