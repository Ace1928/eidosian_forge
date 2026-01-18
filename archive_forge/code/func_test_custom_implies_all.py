import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_custom_implies_all(self):
    self.create_tree()
    out, err = self.run_bzr('version-info --custom --template="{revno} {branch_nick} {clean}\n" branch')
    self.assertEqual('2 branch 1\n', out)
    self.assertEqual('', err)
    self.build_tree_contents([('branch/c', b'now unclean\n')])
    out, err = self.run_bzr('version-info --custom --template="{revno} {branch_nick} {clean}\n" branch')
    self.assertEqual('2 branch 0\n', out)
    self.assertEqual('', err)