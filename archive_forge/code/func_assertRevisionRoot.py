import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def assertRevisionRoot(self, revtree, path):
    self.assertEqual(revtree.get_revision_id(), revtree.get_file_revision(path.decode('utf-8')))