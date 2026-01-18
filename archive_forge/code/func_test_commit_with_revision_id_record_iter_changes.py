import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_commit_with_revision_id_record_iter_changes(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        revision_id = 'Èabc'.encode()
        try:
            try:
                builder = tree.branch.get_commit_builder([], revision_id=revision_id)
            except errors.NonAsciiRevisionId:
                revision_id = b'abc'
                builder = tree.branch.get_commit_builder([], revision_id=revision_id)
        except repository.CannotSetRevisionId:
            return
        self.assertFalse(builder.random_revid)
        try:
            list(builder.record_iter_changes(tree, tree.last_revision(), tree.iter_changes(tree.basis_tree())))
            builder.finish_inventory()
        except:
            builder.abort()
            raise
        self.assertEqual(revision_id, builder.commit('foo bar'))
    self.assertTrue(tree.branch.repository.has_revision(revision_id))
    self.assertEqual(revision_id, tree.branch.repository.revision_tree(revision_id).get_revision_id())