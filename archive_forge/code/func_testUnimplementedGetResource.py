from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testUnimplementedGetResource(self):
    resource_manager = testresources.TestResource()
    self.assertRaises(NotImplementedError, resource_manager.getResource)