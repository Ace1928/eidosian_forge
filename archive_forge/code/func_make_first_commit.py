import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_first_commit(self, repo):
    trunk = repo.controldir.create_branch()
    tree = memorytree.MemoryTree.create_on_branch(trunk)
    tree.lock_write()
    tree.add([''], ['directory'], [b'TREE_ROOT'])
    tree.add(['dir'], ['directory'], [b'dir-id'])
    tree.add(['filename'], ['file'], [b'file-id'])
    tree.put_file_bytes_non_atomic('filename', b'content\n')
    tree.commit('Trunk commit', rev_id=b'rev-0')
    tree.commit('Trunk commit', rev_id=b'rev-1')
    tree.unlock()