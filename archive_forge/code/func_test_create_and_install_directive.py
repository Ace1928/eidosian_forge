from breezy import merge_directive
from breezy.bzr import chk_map
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_create_and_install_directive(self):
    source_branch, target_branch = self.make_two_branches()
    directive = self.create_merge_directive(source_branch, target_branch.base)
    chk_map.clear_cache()
    directive.install_revisions(target_branch.repository)
    rt = target_branch.repository.revision_tree(b'B')
    with rt.lock_read():
        self.assertEqualDiff(b'new content\n', rt.get_file_text('f'))