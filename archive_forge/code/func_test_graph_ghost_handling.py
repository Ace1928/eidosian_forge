import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_graph_ghost_handling(self):
    if not self.repository_format.supports_ghosts:
        raise tests.TestNotApplicable('format does not support ghosts')
    tree = self.make_branch_and_tree('here')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    rev1 = tree.commit('initial commit')
    tree.add_parent_tree_id(b'ghost')
    rev2 = tree.commit('commit-with-ghost')
    graph = tree.branch.repository.get_graph()
    parents = graph.get_parent_map([b'ghost', rev2])
    self.assertTrue(b'ghost' not in parents)
    self.assertEqual(parents[rev2], (rev1, b'ghost'))