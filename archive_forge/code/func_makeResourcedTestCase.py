import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def makeResourcedTestCase(self, has_resource=True):
    case = testresources.ResourcedTestCase('run')
    if has_resource:
        case.resources = [('resource', testresources.TestResource())]
    return case