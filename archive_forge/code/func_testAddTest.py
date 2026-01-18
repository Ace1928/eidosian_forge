import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddTest(self):
    case = self.makeTestCase()
    self.optimising_suite.addTest(case)
    self.assertEqual([case], self.optimising_suite._tests)