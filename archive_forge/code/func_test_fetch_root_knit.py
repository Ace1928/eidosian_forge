from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_root_knit(self):
    """Ensure that knit2.fetch() updates the root knit

        This tests the case where the root has a new revision, but there are no
        corresponding filename, parent, contents or other changes.
        """
    knit1_format = bzrdir.BzrDirMetaFormat1()
    knit1_format.repository_format = knitrepo.RepositoryFormatKnit1()
    knit2_format = bzrdir.BzrDirMetaFormat1()
    knit2_format.repository_format = knitrepo.RepositoryFormatKnit3()
    tree = self.make_branch_and_tree('tree', knit1_format)
    tree.set_root_id(b'tree-root')
    tree.commit('rev1', rev_id=b'rev1')
    tree.commit('rev2', rev_id=b'rev2')
    Convert(tree.basedir, knit2_format)
    tree = WorkingTree.open(tree.basedir)
    branch = self.make_branch('branch', format=knit2_format)
    branch.pull(tree.branch, stop_revision=b'rev1')
    repo = branch.repository
    repo.lock_read()
    try:
        self.assertEqual({(b'tree-root', b'rev1'): ()}, repo.texts.get_parent_map([(b'tree-root', b'rev1'), (b'tree-root', b'rev2')]))
    finally:
        repo.unlock()
    branch.pull(tree.branch)
    repo.lock_read()
    try:
        self.assertEqual({(b'tree-root', b'rev2'): ((b'tree-root', b'rev1'),)}, repo.texts.get_parent_map([(b'tree-root', b'rev2')]))
    finally:
        repo.unlock()