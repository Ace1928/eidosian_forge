import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_from_module_import(self):
    self.assertSetEqual(self.module_gatherer.complete(19, 'from zzefg import a'), {'a1', 'a2'})
    self.assertSetEqual(self.module_gatherer.complete(20, 'from  zzefg import a'), {'a1', 'a2'})
    self.assertSetEqual(self.module_gatherer.complete(20, 'from zzefg  import a'), {'a1', 'a2'})
    self.assertSetEqual(self.module_gatherer.complete(20, 'from zzefg import  a'), {'a1', 'a2'})