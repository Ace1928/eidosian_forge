import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_custom_setUp(self):

    class Stub:

        def mysetup(self):
            self.value = 42
    obj = Stub()
    fixture = fixtures.MethodFixture(obj, setup=obj.mysetup)
    fixture.setUp()
    self.assertEqual(42, fixture.obj.value)
    self.assertEqual(obj, fixture.obj)
    fixture.cleanUp()