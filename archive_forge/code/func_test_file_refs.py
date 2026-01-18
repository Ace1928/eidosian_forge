import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_file_refs(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add('foo')
    revid = tree.commit('a commit')
    out, err = self.run_bzr('file-refs ' + tree.path2id('foo').decode() + ' ' + revid.decode())
    self.assertEqual(out, revid.decode('utf-8') + '\n')
    self.assertEqual(err, '')