import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_from_package(self):
    self.assertSetEqual(self.module_gatherer.complete(17, 'from xml import d'), {'dom'})