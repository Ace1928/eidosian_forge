import testtools
import testresources
from testresources import split_by_resources, _resource_graph
from testresources.tests import ResultWithResourceExtensions
import unittest
def test_discrete(self):
    resset1 = frozenset([testresources.TestResourceManager()])
    resset2 = frozenset([testresources.TestResourceManager()])
    resource_sets = [resset1, resset2]
    result = _resource_graph(resource_sets)
    self.assertEqual({resset1: set([]), resset2: set([])}, result)