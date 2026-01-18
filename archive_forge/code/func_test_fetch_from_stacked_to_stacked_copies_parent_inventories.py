import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
def test_fetch_from_stacked_to_stacked_copies_parent_inventories(self):
    """Fetch from a stacked branch copies inventories for parents of
        revisions at the stacking boundary.

        Specifically, fetch will copy the parent inventories from the
        source for which the corresponding revisions are not present.  This
        will happen even when the source repository has no fallbacks configured
        (as is the case during upgrade).
        """
    if not self.repository_format.supports_external_lookups:
        raise TestNotApplicable('Need stacking support in the source.')
    if not self.repository_format_to.supports_external_lookups:
        raise TestNotApplicable('Need stacking support in the target.')
    builder = self.make_branch_builder('branch')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'base')
    builder.build_snapshot([b'base'], [('modify', ('file', b'left content\n'))], revision_id=b'left')
    builder.build_snapshot([b'base'], [('modify', ('file', b'right content\n'))], revision_id=b'right')
    builder.build_snapshot([b'left', b'right'], [('modify', ('file', b'left and right content\n'))], revision_id=b'merge')
    builder.finish_series()
    branch = builder.get_branch()
    repo = self.make_repository('old-trunk')
    old_trunk = repo.controldir.create_branch()
    old_trunk.repository.fetch(branch.repository, b'left')
    old_trunk.repository.fetch(branch.repository, b'right')
    repo = self.make_to_repository('new-trunk')
    new_trunk = repo.controldir.create_branch()
    new_trunk.repository.fetch(branch.repository, b'left')
    new_trunk.repository.fetch(branch.repository, b'right')
    repo = self.make_repository('old-stacked')
    old_stacked_branch = repo.controldir.create_branch()
    old_stacked_branch.set_stacked_on_url(old_trunk.base)
    old_stacked_branch.repository.fetch(branch.repository, b'merge')
    repo = self.make_to_repository('new-stacked')
    new_stacked_branch = repo.controldir.create_branch()
    new_stacked_branch.set_stacked_on_url(new_trunk.base)
    old_unstacked_repo = old_stacked_branch.controldir.open_repository()
    new_unstacked_repo = new_stacked_branch.controldir.open_repository()
    new_unstacked_repo.fetch(old_unstacked_repo, b'merge')
    new_unstacked_repo.lock_read()
    self.addCleanup(new_unstacked_repo.unlock)
    self.assertFalse(new_unstacked_repo.has_revision(b'left'))
    self.assertFalse(new_unstacked_repo.has_revision(b'right'))
    self.assertEqual({(b'left',), (b'right',), (b'merge',)}, new_unstacked_repo.inventories.keys())
    new_trunk.lock_read()
    self.addCleanup(new_trunk.unlock)
    left_tree, right_tree = new_trunk.repository.revision_trees([b'left', b'right'])
    new_stacked_branch.lock_read()
    self.addCleanup(new_stacked_branch.unlock)
    stacked_left_tree, stacked_right_tree = new_stacked_branch.repository.revision_trees([b'left', b'right'])
    self.assertEqual(left_tree, stacked_left_tree)
    self.assertEqual(right_tree, stacked_right_tree)
    self.assertTrue(new_unstacked_repo.has_revision(b'merge'))
    expected_texts = {(b'file-id', b'merge')}
    if new_stacked_branch.repository.texts.get_parent_map([(b'root-id', b'merge')]):
        expected_texts.add((b'root-id', b'merge'))
    self.assertEqual(expected_texts, new_unstacked_repo.texts.keys())
    self.assertCanStreamRevision(new_unstacked_repo, b'merge')