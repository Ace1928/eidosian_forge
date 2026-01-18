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
def makeRepository(self, root):
    """
        Create a Git repository in the specified path.

        @type root: L{FilePath}
        @params root: The directory to create the Git repository into.

        @return: The path to the repository just created.
        @rtype: L{FilePath}
        """
    _gitInit(root)
    return root