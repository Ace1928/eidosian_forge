import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
def createFiles(self, files, parentDir='.'):
    for filename, contents in self.files:
        filename = os.path.join(parentDir, filename)
        self._createDirectory(filename)
        with open(filename, 'w') as fd:
            fd.write(contents)