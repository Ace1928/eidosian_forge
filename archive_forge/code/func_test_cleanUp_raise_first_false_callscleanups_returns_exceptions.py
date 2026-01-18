import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_cleanUp_raise_first_false_callscleanups_returns_exceptions(self):
    calls = []

    def raise_exception1():
        calls.append('1')
        raise Exception('woo')

    def raise_exception2():
        calls.append('2')
        raise Exception('woo')

    class FixtureWithException(fixtures.Fixture):

        def setUp(self):
            super(FixtureWithException, self).setUp()
            self.addCleanup(raise_exception2)
            self.addCleanup(raise_exception1)
    fixture = FixtureWithException()
    fixture.setUp()
    exceptions = fixture.cleanUp(raise_first=False)
    self.assertEqual(['1', '2'], calls)
    self.assertEqual(2, len(exceptions))
    self.assertEqual(3, len(exceptions[0]))
    type, value, tb = exceptions[0]
    self.assertEqual(Exception, type)
    self.assertIsInstance(value, Exception)
    self.assertEqual(('woo',), value.args)
    self.assertIsInstance(tb, types.TracebackType)