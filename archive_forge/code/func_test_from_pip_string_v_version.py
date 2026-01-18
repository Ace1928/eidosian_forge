import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_v_version(self):
    parsed = from_pip_string('v1.2.3')
    expected = version.SemanticVersion(1, 2, 3)
    self.expectThat(expected, matchers.Equals(parsed))
    expected = version.SemanticVersion(1, 2, 3, 'a', 5, dev_count=6)
    parsed = from_pip_string('V1.2.3.0a4.post6')
    self.expectThat(expected, matchers.Equals(parsed))
    self.expectThat(lambda: from_pip_string('x1.2.3'), matchers.raises(ValueError))