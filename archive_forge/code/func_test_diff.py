import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff(self):
    tree = self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!')])
    tree.commit(message='fixing hello')
    output = self.run_bzr('diff -r 2..3', retcode=1)[0]
    self.assertTrue('\n+hello world!' in output)
    output = self.run_bzr('diff -c 3', retcode=1)[0]
    self.assertTrue('\n+hello world!' in output)
    output = self.run_bzr('diff -r last:3..last:1', retcode=1)[0]
    self.assertTrue('\n+baz' in output)
    output = self.run_bzr('diff -c last:2', retcode=1)[0]
    self.assertTrue('\n+baz' in output)
    self.build_tree(['moo'])
    tree.add('moo')
    os.unlink('moo')
    self.run_bzr('diff')