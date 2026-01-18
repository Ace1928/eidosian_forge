import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_from(self):
    base_tree = self.make_branch_and_tree('base')
    self.build_tree(['base/a', 'base/b/', 'base/b/c'])
    base_tree.add(['a', 'b', 'b/c'])
    base_tree.commit('foo')
    new_tree = self.make_branch_and_tree('new')
    self.build_tree(['new/a', 'new/b/', 'new/b/c', 'd'])
    out, err = self.run_bzr('add --file-ids-from ../base', working_dir='new')
    self.assertEqual('', err)
    self.assertEqualDiff('adding a w/ file id from a\nadding b w/ file id from b\nadding b/c w/ file id from b/c\n', out)
    new_tree = new_tree.controldir.open_workingtree()
    self.assertEqual(base_tree.path2id('a'), new_tree.path2id('a'))
    self.assertEqual(base_tree.path2id('b'), new_tree.path2id('b'))
    self.assertEqual(base_tree.path2id('b/c'), new_tree.path2id('b/c'))