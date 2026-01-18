import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
@unittest.expectedFailure
def test_import_blank(self):
    self.assertSetEqual(self.module_gatherer.complete(7, 'import '), {'zzabc', 'zzabd', 'zzefg'})
    self.assertSetEqual(self.module_gatherer.complete(8, 'import  '), {'zzabc', 'zzabd', 'zzefg'})