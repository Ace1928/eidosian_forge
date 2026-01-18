import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
def prepare_export(self):
    work_a = self.make_branch_and_tree('wta')
    self.build_tree_contents([('wta/file', b'a\nb\nc\nd\n'), ('wta/dir', b'')])
    work_a.add('file')
    work_a.add('dir')
    work_a.commit('add file')
    tree_a = self.workingtree_to_test_tree(work_a)
    export(tree_a, 'output', self.exporter)