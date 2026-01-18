import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_legacy_no_0_prerelease_2(self):
    expected = version.SemanticVersion(2, 0, 0, prerelease_type='rc', prerelease=1)
    parsed = from_pip_string('2.0.0.rc1')
    self.assertEqual(expected, parsed)