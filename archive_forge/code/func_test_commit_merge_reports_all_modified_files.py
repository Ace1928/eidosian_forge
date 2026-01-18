import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_merge_reports_all_modified_files(self):
    this_tree = self.make_branch_and_tree('this')
    self.build_tree(['this/dirtorename/', 'this/dirtoreparent/', 'this/dirtoleave/', 'this/dirtoremove/', 'this/filetoreparent', 'this/filetorename', 'this/filetomodify', 'this/filetoremove', 'this/filetoleave'])
    this_tree.add(['dirtorename', 'dirtoreparent', 'dirtoleave', 'dirtoremove', 'filetoreparent', 'filetorename', 'filetomodify', 'filetoremove', 'filetoleave'])
    this_tree.commit('create_files')
    other_dir = this_tree.controldir.sprout('other')
    other_tree = other_dir.open_workingtree()
    with other_tree.lock_write():
        other_tree.rename_one('dirtorename', 'renameddir')
        other_tree.rename_one('dirtoreparent', 'renameddir/reparenteddir')
        other_tree.rename_one('filetorename', 'renamedfile')
        other_tree.rename_one('filetoreparent', 'renameddir/reparentedfile')
        other_tree.remove(['dirtoremove', 'filetoremove'])
        self.build_tree_contents([('other/newdir/',), ('other/filetomodify', b'new content'), ('other/newfile', b'new file content')])
        other_tree.add('newfile')
        other_tree.add('newdir/')
        other_tree.commit('modify all sample files and dirs.')
    this_tree.merge_from_branch(other_tree.branch)
    out, err = self.run_bzr('commit -m added', working_dir='this')
    self.assertEqual('', out)
    self.assertEqual({'Committing to: %s/' % osutils.pathjoin(osutils.getcwd(), 'this'), 'modified filetomodify', 'added newdir', 'added newfile', 'renamed dirtorename => renameddir', 'renamed filetorename => renamedfile', 'renamed dirtoreparent => renameddir/reparenteddir', 'renamed filetoreparent => renameddir/reparentedfile', 'deleted dirtoremove', 'deleted filetoremove', 'Committed revision 2.', ''}, set(err.split('\n')))