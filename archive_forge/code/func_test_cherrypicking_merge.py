import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_cherrypicking_merge(self):
    source = self.make_branch_and_tree('source')
    for f in ('a', 'b', 'c', 'd'):
        self.build_tree(['source/' + f])
        source.add(f)
        source.commit('added ' + f, rev_id=b'rev_' + f.encode('ascii'))
    target = source.controldir.sprout('target', b'rev_a').open_workingtree()
    self.assertDirectoryContent('target', ['.bzr', 'a'])
    self.run_bzr('merge -d target -r revid:rev_b..revid:rev_c source')
    self.assertDirectoryContent('target', ['.bzr', 'a', 'c'])
    target.revert()
    self.run_bzr('merge -d target -r revid:rev_b..revid:rev_d source')
    self.assertDirectoryContent('target', ['.bzr', 'a', 'c', 'd'])
    target.revert()
    self.run_bzr('merge -d target -c revid:rev_d source')
    self.assertDirectoryContent('target', ['.bzr', 'a', 'd'])