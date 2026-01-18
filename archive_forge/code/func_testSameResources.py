import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSameResources(self):
    a = self.makeResource()
    b = self.makeResource()
    self.assertEqual(0, self.suite.cost_of_switching(set([a]), set([a])))
    self.assertEqual(0, self.suite.cost_of_switching(set([a, b]), set([a, b])))