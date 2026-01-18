import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_merge_bundle(self):
    self.prepare_merge_directive()
    self.tree1.commit('baz', rev_id=b'baz-id')
    md_text = self.run_bzr(['merge-directive', self.tree2.basedir, '-r', '2', '/dev/null', '--bundle'])[0]
    self.build_tree_contents([('../directive', md_text)])
    os.chdir('../tree2')
    self.run_bzr('merge ../directive')
    wt = workingtree.WorkingTree.open('.')
    self.assertEqual(b'bar-id', wt.get_parent_ids()[1])