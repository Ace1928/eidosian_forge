import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
def test_duplicate_details_are_disambiguated(self):
    parent = fixtures.Fixture()
    with parent:
        parent.addDetail('foo', 'parent-content')
        child = fixtures.Fixture()
        parent.useFixture(child)
        child.addDetail('foo', 'child-content')
        self.assertEqual({'foo': 'parent-content', 'foo-1': 'child-content'}, parent.getDetails())