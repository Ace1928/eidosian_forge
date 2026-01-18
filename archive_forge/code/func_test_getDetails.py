import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_getDetails(self):
    fixture = fixtures.Fixture()
    with fixture:
        self.assertEqual({}, fixture.getDetails())