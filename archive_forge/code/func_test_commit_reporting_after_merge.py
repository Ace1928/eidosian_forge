import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_commit_reporting_after_merge(self):
    this_tree = self.make_branch_and_tree('this')
    self.build_tree(['this/dirtorename/', 'this/dirtoreparent/', 'this/dirtoleave/', 'this/dirtoremove/', 'this/filetoreparent', 'this/filetorename', 'this/filetomodify', 'this/filetoremove', 'this/filetoleave'])
    this_tree.add(['dirtorename', 'dirtoreparent', 'dirtoleave', 'dirtoremove', 'filetoreparent', 'filetorename', 'filetomodify', 'filetoremove', 'filetoleave'])
    this_tree.commit('create_files')
    other_dir = this_tree.controldir.sprout('other')
    other_tree = other_dir.open_workingtree()
    other_tree.lock_write()
    try:
        other_tree.rename_one('dirtorename', 'renameddir')
        other_tree.rename_one('dirtoreparent', 'renameddir/reparenteddir')
        other_tree.rename_one('filetorename', 'renamedfile')
        other_tree.rename_one('filetoreparent', 'renameddir/reparentedfile')
        other_tree.remove(['dirtoremove', 'filetoremove'])
        self.build_tree_contents([('other/newdir/',), ('other/filetomodify', b'new content'), ('other/newfile', b'new file content')])
        other_tree.add('newfile')
        other_tree.add('newdir/')
        other_tree.commit('modify all sample files and dirs.')
    finally:
        other_tree.unlock()
    this_tree.merge_from_branch(other_tree.branch)
    reporter = CapturingReporter()
    this_tree.commit('do the commit', reporter=reporter)
    expected = {('change', 'modified', 'filetomodify'), ('change', 'added', 'newdir'), ('change', 'added', 'newfile'), ('renamed', 'renamed', 'dirtorename', 'renameddir'), ('renamed', 'renamed', 'filetorename', 'renamedfile'), ('renamed', 'renamed', 'dirtoreparent', 'renameddir/reparenteddir'), ('renamed', 'renamed', 'filetoreparent', 'renameddir/reparentedfile'), ('deleted', 'dirtoremove'), ('deleted', 'filetoremove')}
    result = set(reporter.calls)
    missing = expected - result
    new = result - expected
    self.assertEqual((set(), set()), (missing, new))