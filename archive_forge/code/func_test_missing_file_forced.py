from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_missing_file_forced(self):
    tree = self.make_branch_and_tree('tree')
    tree.add_reference(self.make_branch_and_tree('tree/file'))
    out, err = self.run_bzr('reference --force-unversioned file http://example.org', working_dir='tree')
    location = tree.get_reference_info('file')
    self.assertEqual('http://example.org', location)
    self.assertEqual('', out)
    self.assertEqual('', err)