import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
def test_allocate(self):

    def allocate():
        '.' * BIG_FILE_SIZE
    self.assertRaises(MemoryError, allocate)