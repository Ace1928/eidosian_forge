import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testSetUpUsesSuper(self):

    class OtherBaseCase(unittest.TestCase):
        setUpCalled = False

        def setUp(self):
            self.setUpCalled = True
            super(OtherBaseCase, self).setUp()

    class OurCase(testresources.ResourcedTestCase, OtherBaseCase):

        def runTest(self):
            pass
    ourCase = OurCase()
    ourCase.setUp()
    self.assertTrue(ourCase.setUpCalled)