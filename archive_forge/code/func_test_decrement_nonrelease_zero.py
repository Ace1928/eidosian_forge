import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_decrement_nonrelease_zero(self):
    semver = version.SemanticVersion(1, 0, 0)
    self.assertEqual(version.SemanticVersion(0, 9999, 9999), semver.decrement())