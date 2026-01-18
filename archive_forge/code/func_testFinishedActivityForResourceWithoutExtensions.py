from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testFinishedActivityForResourceWithoutExtensions(self):
    result = ResultWithoutResourceExtensions()
    resource_manager = MockResource()
    r = resource_manager.getResource()
    resource_manager.finishedWith(r, result)