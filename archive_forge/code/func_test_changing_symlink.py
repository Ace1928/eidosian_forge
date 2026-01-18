import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
def test_changing_symlink(self):
    self.make_git_repo('d')
    os.chdir('d')
    bb = GitBranchBuilder()
    bb.set_symlink('mylink', 'target')
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
    bb.set_symlink('mylink', 'target/')
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
    marks = bb.finish()
    gitsha1 = marks[mark1]
    gitsha2 = marks[mark2]
    os.chdir('..')
    oldrepo = self.open_git_repo('d')
    newrepo = self.clone_git_repo('d', 'f')
    revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
    revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    tree1 = newrepo.revision_tree(revid1)
    tree2 = newrepo.revision_tree(revid2)
    self.assertEqual(revid1, tree1.get_file_revision('mylink'))
    self.assertEqual('target', tree1.get_symlink_target('mylink'))
    self.assertEqual(revid2, tree2.get_file_revision('mylink'))
    self.assertEqual('target/', tree2.get_symlink_target('mylink'))