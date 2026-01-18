import os
import shutil
from .... import bedding, config, errors, trace
from ....merge import Merger
from ....mutabletree import MutableTree
from ....tests import TestCaseWithTransport, TestSkipped
from .. import (post_build_tree_quilt, post_merge_quilt_cleanup,
from ..merge import tree_unapply_patches
from ..quilt import QuiltPatches
from . import quilt_feature
def test_no_patches(self):
    tree = self.make_branch_and_tree('.')
    new_tree, target_dir = tree_unapply_patches(tree)
    self.assertIs(tree, new_tree)
    self.assertIs(None, target_dir)