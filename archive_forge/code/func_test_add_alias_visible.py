from breezy import config, tests
from breezy.tests import features
def test_add_alias_visible(self):
    """Adding an alias makes it ..."""
    self.run_bzr('alias commit="commit --strict"')
    out, err = self.run_bzr('alias commit')
    self.assertEqual('brz alias commit="commit --strict"\n', out)