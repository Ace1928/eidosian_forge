from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDefaultCosts(self):
    resource_manager = testresources.TestResource()
    self.assertEqual(resource_manager.setUpCost, 1)
    self.assertEqual(resource_manager.tearDownCost, 1)