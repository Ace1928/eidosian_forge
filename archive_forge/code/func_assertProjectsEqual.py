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
def assertProjectsEqual(self, observedProjects, expectedProjects):
    """
        Assert that two lists of L{Project}s are equal.
        """
    self.assertEqual(len(observedProjects), len(expectedProjects))
    observedProjects = sorted(observedProjects, key=operator.attrgetter('directory'))
    expectedProjects = sorted(expectedProjects, key=operator.attrgetter('directory'))
    for observed, expected in zip(observedProjects, expectedProjects):
        self.assertEqual(observed.directory, expected.directory)