import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def make_abcd_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/c'])
    tree.add(['a', 'c'])
    tree.commit('record old names')
    osutils.rename('tree/a', 'tree/b')
    osutils.rename('tree/c', 'tree/d')
    return tree