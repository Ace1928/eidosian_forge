from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetActivityForResourceWithExtensions(self):
    result = ResultWithResourceExtensions()
    resource_manager = MockResource()
    r = resource_manager.getResource(result)
    expected = [('make', 'start', resource_manager), ('make', 'stop', resource_manager)]
    resource_manager.finishedWith(r)
    self.assertEqual(expected, result._calls)