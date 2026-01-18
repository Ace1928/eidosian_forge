import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
def test_mock_patch_object_with_replacement(self):
    self.useFixture(MockPatchObject(Foo, 'bar', mocking_bar))
    instance = Foo()
    self.assertEqual(instance.bar(), 'mocked!')