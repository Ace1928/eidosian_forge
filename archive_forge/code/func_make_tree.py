import tarfile
import zipfile
from .. import export, filter_tree, tests
from . import fixtures
from .test_filters import _stack_1
def make_tree(self):
    self.underlying_tree = fixtures.make_branch_and_populated_tree(self)

    def stack_callback(path):
        return _stack_1
    self.filter_tree = filter_tree.ContentFilterTree(self.underlying_tree, stack_callback)
    return self.filter_tree