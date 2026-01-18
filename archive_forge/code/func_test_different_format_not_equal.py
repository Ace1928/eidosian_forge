from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_different_format_not_equal(self):
    """Different format repositories are comparable and not the same.

        Comparing different format repository objects should give a negative
        result, rather than trigger an exception (which could happen with a
        naive __eq__ implementation, e.g. due to missing attributes).
        """
    repo = self.make_repository('repo')
    other_repo = self.make_repository('other', format='default')
    if repo._format == other_repo._format:
        transport.get_transport_from_url(self.get_vfs_only_url()).delete_tree('other')
        other_repo = self.make_repository('other', format='knit')
    other_bzrdir = controldir.ControlDir.open(self.get_vfs_only_url('other'))
    other_repo = other_bzrdir.open_repository()
    self.assertDifferentRepo(repo, other_repo)