from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def test_default_uses_setUp_tearDown(self):
    calls = []

    class Wrapped:

        def setUp(self):
            calls.append('setUp')

        def tearDown(self):
            calls.append('tearDown')
    mgr = testresources.GenericResource(Wrapped)
    resource = mgr.getResource()
    self.assertEqual(['setUp'], calls)
    mgr.finishedWith(resource)
    self.assertEqual(['setUp', 'tearDown'], calls)
    self.assertIsInstance(resource, Wrapped)