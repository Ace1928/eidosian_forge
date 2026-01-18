import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def create_base(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', b'foo\n')])
    wt.add('a')
    wt.commit('adding a')
    self.build_tree_contents([('b', b'non-ascii \xff\xff\xfc\xfb\x00 in b\n')])
    wt.add('b')
    wt.commit(self.info['message'])
    fname = self.info['filename']
    self.build_tree_contents([(fname, b'unicode filename\n')])
    wt.add(fname)
    wt.commit('And a unicode file\n')
    self.wt = wt