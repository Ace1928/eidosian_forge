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
def assertStructure(self, root, dirDict):
    """
        Assert that a directory is equivalent to one described by a dict.

        @param root: The filesystem directory to compare.
        @type root: L{FilePath}
        @param dirDict: The dict that should describe the contents of the
            directory. It should be the same structure as the C{dirDict}
            parameter to L{createStructure}.
        @type dirDict: C{dict}
        """
    children = [each.basename() for each in root.children()]
    for pathSegment, expectation in dirDict.items():
        child = root.child(pathSegment)
        if callable(expectation):
            self.assertTrue(expectation(child))
        elif isinstance(expectation, dict):
            self.assertTrue(child.isdir(), f'{child.path} is not a dir!')
            self.assertStructure(child, expectation)
        else:
            actual = child.getContent().decode().replace(os.linesep, '\n')
            self.assertEqual(actual, expectation)
        children.remove(pathSegment)
    if children:
        self.fail(f'There were extra children in {root.path}: {children}')