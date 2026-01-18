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
def test_fetch_funky_file_id(self):
    from_tree = self.make_branch_and_tree('tree')
    if sys.platform == 'win32':
        from_repo = from_tree.branch.repository
        check_repo_format_for_funky_id_on_win32(from_repo)
    self.build_tree(['tree/filename'])
    if not from_tree.supports_setting_file_ids():
        raise TestNotApplicable('from tree format can not create custom file ids')
    from_tree.add('filename', ids=b'funky-chars<>%&;"\'')
    from_tree.commit('commit filename')
    to_repo = self.make_to_repository('to')
    try:
        to_repo.fetch(from_tree.branch.repository, from_tree.get_parent_ids()[0])
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')