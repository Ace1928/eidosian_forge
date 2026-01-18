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
def test_fetch_from_stacked(self):
    """Fetch from a stacked branch succeeds."""
    if not self.repository_format.supports_external_lookups:
        raise TestNotApplicable('Need stacking support in the source.')
    builder = self.make_branch_builder('full-branch')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'first')
    builder.build_snapshot([b'first'], [('modify', ('file', b'second content\n'))], revision_id=b'second')
    builder.build_snapshot([b'second'], [('modify', ('file', b'third content\n'))], revision_id=b'third')
    builder.finish_series()
    branch = builder.get_branch()
    repo = self.make_repository('stacking-base')
    trunk = repo.controldir.create_branch()
    trunk.repository.fetch(branch.repository, b'second')
    repo = self.make_repository('stacked')
    stacked_branch = repo.controldir.create_branch()
    stacked_branch.set_stacked_on_url(trunk.base)
    stacked_branch.repository.fetch(branch.repository, b'third')
    target = self.make_to_repository('target')
    try:
        target.fetch(stacked_branch.repository, b'third')
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    target.lock_read()
    self.addCleanup(target.unlock)
    all_revs = {b'first', b'second', b'third'}
    self.assertEqual(all_revs, set(target.get_parent_map(all_revs)))