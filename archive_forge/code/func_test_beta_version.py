import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_beta_version(self):
    semver = version.SemanticVersion(1, 2, 4, 'b', 1)
    self.assertEqual((1, 2, 4, 'beta', 1), semver.version_tuple())
    self.assertEqual('1.2.4', semver.brief_string())
    self.assertEqual('1.2.4~b1', semver.debian_string())
    self.assertEqual('1.2.4.0b1', semver.release_string())
    self.assertEqual('1.2.3.b1', semver.rpm_string())
    self.assertEqual(semver, from_pip_string('1.2.4.0b1'))