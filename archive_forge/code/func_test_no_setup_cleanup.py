import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_no_setup_cleanup(self):

    class Stub:
        pass
    fixture = fixtures.MethodFixture(Stub())
    fixture.setUp()
    fixture.reset()
    self.assertIsInstance(fixture.obj, Stub)
    fixture.cleanUp()