import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_backslash(self):
    if os.path.sep == '\\':
        raise tests.TestNotApplicable('unable to add filenames with backslashes where  it is the path separator')
    tree = self.make_branch_and_tree('.')
    self.build_tree(['\\'])
    self.assertEqual('adding \\\n', self.run_bzr('add \\\\')[0])
    self.assertEqual('\\\n', self.run_bzr('ls --versioned')[0])