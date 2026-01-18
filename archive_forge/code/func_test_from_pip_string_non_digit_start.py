import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_non_digit_start(self):
    self.assertRaises(ValueError, from_pip_string, 'non-release-tag/2014.12.16-1')