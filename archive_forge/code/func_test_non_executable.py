import os
import sys
import tempfile
from .. import mergetools, tests
def test_non_executable(self):
    f, name = tempfile.mkstemp()
    try:
        self.log('temp filename: %s', name)
        self.assertFalse(mergetools.check_availability(name))
    finally:
        os.close(f)
        os.unlink(name)