import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
def test_mock_multiple_with_replacement(self):
    self.useFixture(MockPatchMultiple('%s.Foo' % __name__, bar=mocking_bar))
    instance = Foo()
    self.assertEqual(instance.bar(), 'mocked!')