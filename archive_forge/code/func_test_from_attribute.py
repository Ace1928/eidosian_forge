import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_from_attribute(self):
    self.assertSetEqual(self.module_gatherer.complete(19, 'from sys import arg'), {'argv'})