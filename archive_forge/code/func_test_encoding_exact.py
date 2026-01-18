import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_encoding_exact(self):
    tree1, tree2 = self.prepare_merge_directive()
    tree1.commit('messagé')
    self.run_bzr('merge-directive ../tree2')