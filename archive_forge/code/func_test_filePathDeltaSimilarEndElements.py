import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_filePathDeltaSimilarEndElements(self):
    """
        L{filePathDelta} doesn't take into account final elements when
        comparing 2 paths, but stops at the first difference.
        """
    self.assertEqual(filePathDelta(FilePath('/foo/bar/bar/spam'), FilePath('/foo/bar/baz/spam')), ['..', '..', 'baz', 'spam'])