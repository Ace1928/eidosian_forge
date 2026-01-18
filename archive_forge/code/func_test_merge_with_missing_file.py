import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_with_missing_file(self):
    """Merge handles missing file conflicts"""
    self.build_tree_contents([('a/',), ('a/sub/',), ('a/sub/a.txt', b'hello\n'), ('a/b.txt', b'hello\n'), ('a/sub/c.txt', b'hello\n')])
    a_tree = self.make_branch_and_tree('a')
    a_tree.add(['sub', 'b.txt', 'sub/c.txt', 'sub/a.txt'])
    a_tree.commit(message='added a')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('a/sub/a.txt', b'hello\nthere\n'), ('a/b.txt', b'hello\nthere\n'), ('a/sub/c.txt', b'hello\nthere\n')])
    a_tree.commit(message='Added there')
    os.remove('a/sub/a.txt')
    os.remove('a/sub/c.txt')
    os.rmdir('a/sub')
    os.remove('a/b.txt')
    a_tree.commit(message='Removed a.txt')
    self.build_tree_contents([('b/sub/a.txt', b'hello\nsomething\n'), ('b/b.txt', b'hello\nsomething\n'), ('b/sub/c.txt', b'hello\nsomething\n')])
    b_tree.commit(message='Modified a.txt')
    self.run_bzr('merge ../a/', retcode=1, working_dir='b')
    self.assertPathExists('b/sub/a.txt.THIS')
    self.assertPathExists('b/sub/a.txt.BASE')
    self.run_bzr('merge ../b/', retcode=1, working_dir='a')
    self.assertPathExists('a/sub/a.txt.OTHER')
    self.assertPathExists('a/sub/a.txt.BASE')