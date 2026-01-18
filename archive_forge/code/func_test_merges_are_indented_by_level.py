import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_merges_are_indented_by_level(self):
    self.run_bzr(['log', '-n0'], working_dir='level0')
    revnos_and_depth = [(r.revno, r.merge_depth) for r in self.get_captured_revisions()]
    self.assertEqual([('2', 0), ('1.1.2', 1), ('1.2.1', 2), ('1.1.1', 1), ('1', 0)], [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])