import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test__setUp_fails_cleanUp_fails(self):

    class Subclass(fixtures.Fixture):

        def _setUp(self):
            self.addDetail('log', text_content('stuff'))
            self.addCleanup(lambda: 1 / 0)
            raise Exception('fred')
    f = Subclass()
    e = self.assertRaises(fixtures.MultipleExceptions, f.setUp)
    self.assertRaises(TypeError, f.cleanUp)
    self.assertEqual(Exception, e.args[0][0])
    self.assertEqual(ZeroDivisionError, e.args[1][0])
    self.assertEqual(fixtures.SetupError, e.args[2][0])
    self.assertEqual('stuff', e.args[2][1].args[0]['log'].as_text())