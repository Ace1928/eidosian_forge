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
def test_auto_apply_patches_after_checkout(self):
    self.enable_hooks()
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/debian/', 'a/debian/patches/'])
    self.build_tree_contents([('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
    tree_a.smart_add([tree_a.basedir])
    tree_a.commit('initial')
    bedding.ensure_config_dir_exists()
    config.GlobalStack().set('quilt.tree_policy', 'applied')
    tree_a.branch.create_checkout('b')
    self.assertFileEqual('a\n', 'b/a')