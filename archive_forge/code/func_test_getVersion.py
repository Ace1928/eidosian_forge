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
def test_getVersion(self):
    """
        Project objects know their version.
        """
    version = ('twisted', 2, 1, 0)
    project = self.makeProject(version)
    self.assertEqual(project.getVersion(), Version(*version))