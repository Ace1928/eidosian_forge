import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_decrement_nonrelease(self):
    semver = version.SemanticVersion(1, 2, 4, 'b', 1)
    self.assertEqual(version.SemanticVersion(1, 2, 3), semver.decrement())