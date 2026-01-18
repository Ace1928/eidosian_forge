import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def assertDirectoryContent(self, directory, entries, message=''):
    """Assert whether entries (file or directories) exist in a directory.

        It also checks that there are no extra entries.
        """
    ondisk = os.listdir(directory)
    if set(ondisk) == set(entries):
        return
    if message:
        message += '\n'
    raise AssertionError('%s"%s" directory content is different:\na = %s\nb = %s\n' % (message, directory, sorted(entries), sorted(ondisk)))