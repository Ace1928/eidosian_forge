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
def makeProjects(self, *versions):
    """
        Create a series of projects underneath a temporary base directory.

        @return: A L{FilePath} for the base directory.
        """
    baseDirectory = FilePath(self.mktemp())
    for version in versions:
        self.makeProject(version, baseDirectory)
    return baseDirectory