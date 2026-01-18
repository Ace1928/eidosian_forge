from breezy import config, tests
from breezy.tests import features
def test_list_unknown_alias(self):
    out, err = self.run_bzr('alias commit')
    self.assertEqual('brz alias: commit: not found\n', out)