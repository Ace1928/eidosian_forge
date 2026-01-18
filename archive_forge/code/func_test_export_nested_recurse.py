import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
def test_export_nested_recurse(self):
    self.prepare_nested_export(True)
    names = self.get_export_names()
    self.assertIn('output/subdir/b', names)