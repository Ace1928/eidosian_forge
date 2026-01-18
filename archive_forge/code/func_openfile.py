import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def openfile(self, fname, mode):
    """
        This is a mock for L{open}.  It keeps track of opened files so extra
        descriptors can be returned from the mock for L{os.listdir} when used on
        one of the list-of-filedescriptors directories.

        A L{FakeFile} is returned which can be closed to remove the new
        descriptor from the open list.
        """
    f = FakeFile(self, min(set(range(1024)) - set(self._files)))
    self._files.append(f.fd)
    return f