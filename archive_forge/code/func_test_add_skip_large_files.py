import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_skip_large_files(self):
    """Test skipping files larger than add.maximum_file_size"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['small.txt', 'big.txt', 'big2.txt'])
    self.build_tree_contents([('small.txt', b'0\n')])
    self.build_tree_contents([('big.txt', b'01234567890123456789\n')])
    self.build_tree_contents([('big2.txt', b'01234567890123456789\n')])
    tree.branch.get_config_stack().set('add.maximum_file_size', 5)
    out = self.run_bzr('add')[0]
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual(['adding small.txt'], results)
    out, err = self.run_bzr(['add', 'big2.txt'])
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual(['adding big2.txt'], results)
    self.assertEqual('', err)
    tree.branch.get_config_stack().set('add.maximum_file_size', 30)
    out = self.run_bzr('add')[0]
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual(['adding big.txt'], results)