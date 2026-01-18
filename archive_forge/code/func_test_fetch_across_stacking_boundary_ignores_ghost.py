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
def test_fetch_across_stacking_boundary_ignores_ghost(self):
    if not self.repository_format_to.supports_external_lookups:
        raise TestNotApplicable('Need stacking support in the target.')
    if not self.repository_format.supports_ghosts:
        raise TestNotApplicable('Need ghost support in the source.')
    to_repo = self.make_to_repository('to')
    builder = self.make_branch_builder('branch')
    builder.start_series()
    base = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('file', None, 'file', b'content\n'))])
    second = builder.build_snapshot([base], [('modify', ('file', b'second content\n'))])
    third = builder.build_snapshot([second, b'ghost'], [('modify', ('file', b'third content\n'))])
    builder.finish_series()
    branch = builder.get_branch()
    revtree = branch.repository.revision_tree(base)
    root_id = revtree.path2id('')
    file_id = revtree.path2id('file')
    repo = self.make_to_repository('trunk')
    trunk = repo.controldir.create_branch()
    trunk.repository.fetch(branch.repository, second)
    repo = self.make_to_repository('stacked')
    stacked_branch = repo.controldir.create_branch()
    stacked_branch.set_stacked_on_url(trunk.base)
    stacked_branch.repository.fetch(branch.repository, third)
    unstacked_repo = stacked_branch.controldir.open_repository()
    unstacked_repo.lock_read()
    self.addCleanup(unstacked_repo.unlock)
    self.assertFalse(unstacked_repo.has_revision(second))
    self.assertFalse(unstacked_repo.has_revision(b'ghost'))
    self.assertEqual({(second,), (third,)}, unstacked_repo.inventories.keys())
    trunk.lock_read()
    self.addCleanup(trunk.unlock)
    second_tree = trunk.repository.revision_tree(second)
    stacked_branch.lock_read()
    self.addCleanup(stacked_branch.unlock)
    stacked_second_tree = stacked_branch.repository.revision_tree(second)
    self.assertEqual(second_tree, stacked_second_tree)
    self.assertTrue(unstacked_repo.has_revision(third))
    expected_texts = {(file_id, third)}
    if stacked_branch.repository.texts.get_parent_map([(root_id, third)]):
        expected_texts.add((root_id, third))
    self.assertEqual(expected_texts, unstacked_repo.texts.keys())
    self.assertCanStreamRevision(unstacked_repo, third)