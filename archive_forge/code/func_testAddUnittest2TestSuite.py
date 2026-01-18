import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
@testtools.skipIf(unittest2 is None, 'Unittest2 needed')
def testAddUnittest2TestSuite(self):
    case = self.makeTestCase()
    suite = unittest2.TestSuite([case])
    self.optimising_suite.addTest(suite)
    self.assertEqual([case], self.optimising_suite._tests)