import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
def prepare_symlink_export(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    work_a = self.make_branch_and_tree('wta')
    os.symlink('target', 'wta/link')
    work_a.add('link')
    work_a.commit('add link')
    tree_a = self.workingtree_to_test_tree(work_a)
    export(tree_a, 'output', self.exporter)