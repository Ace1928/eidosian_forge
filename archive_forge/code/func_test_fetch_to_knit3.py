from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_knit3(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/foo'])
    tree_a.add('foo')
    rev1 = tree_a.commit('rev1')
    f = controldir.format_registry.make_controldir('development-subtree')
    try:
        format = tree_a.branch.repository._format
        format.check_conversion_target(f.repository_format)
    except errors.BadConversionTarget as e:
        raise TestSkipped(str(e))
    self.get_transport().mkdir('b')
    b_bzrdir = f.initialize(self.get_url('b'))
    knit3_repo = b_bzrdir.create_repository()
    knit3_repo.fetch(tree_a.branch.repository, revision_id=None)
    knit3_repo = b_bzrdir.open_repository()
    rev1_tree = knit3_repo.revision_tree(rev1)
    with rev1_tree.lock_read():
        lines = rev1_tree.get_file_lines('')
    self.assertEqual([], lines)
    b_branch = b_bzrdir.create_branch()
    b_branch.pull(tree_a.branch)
    try:
        tree_b = b_bzrdir.create_workingtree()
    except errors.NotLocalUrl:
        try:
            tree_b = b_branch.create_checkout('b', lightweight=True)
        except errors.NotLocalUrl:
            raise TestSkipped('cannot make working tree with transport %r' % b_bzrdir.transport)
    rev2 = tree_b.commit('no change')
    rev2_tree = knit3_repo.revision_tree(rev2)
    self.assertEqual(rev1, rev2_tree.get_file_revision(''))