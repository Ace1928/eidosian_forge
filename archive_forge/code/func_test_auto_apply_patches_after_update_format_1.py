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
def test_auto_apply_patches_after_update_format_1(self):
    self.enable_hooks()
    tree_a = self.make_branch_and_tree('a')
    tree_b = tree_a.branch.create_checkout('b')
    self.build_tree(['a/debian/', 'a/debian/patches/', 'a/.pc/'])
    self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
    tree_a.smart_add([tree_a.basedir])
    tree_a.commit('initial')
    self.build_tree(['b/.bzr-builddeb/', 'b/debian/', 'b/debian/source/'])
    tree_b.get_config_stack().set('quilt.tree_policy', 'applied')
    self.build_tree_contents([('b/debian/source/format', '1.0')])
    tree_b.update()
    self.assertFileEqual('a\n', 'b/a')