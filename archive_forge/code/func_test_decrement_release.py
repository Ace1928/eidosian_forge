import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_decrement_release(self):
    semver = version.SemanticVersion(2, 2, 5)
    self.assertEqual(version.SemanticVersion(2, 2, 4), semver.decrement())