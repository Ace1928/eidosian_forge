import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_conflicting_continue(self):
    self.make_file('hello', '42')
    self.run_bzr('commit -m that')
    os.chdir('../feature')
    self.make_file('hello', 'other data')
    self.run_bzr('commit -m this')
    self.run_bzr_error(["Text conflict in hello\n1 conflicts encountered.\nbrz: ERROR: A conflict occurred replaying a commit. Resolve the conflict and run 'brz rebase-continue' or run 'brz rebase-abort'.\n"], ['rebase', '../main'])
    self.run_bzr('resolved hello')
    self.assertEqual('', self.run_bzr('rebase-continue')[0])
    self.assertEqual('3\n', self.run_bzr('revno')[0])