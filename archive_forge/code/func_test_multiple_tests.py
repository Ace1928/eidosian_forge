import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_multiple_tests(self):

    class Reference1(unittest.TestCase):
        scenarios = [('1', {}), ('2', {})]

        def test_something(self):
            pass

    class Reference2(unittest.TestCase):
        scenarios = [('3', {}), ('4', {})]

        def test_something(self):
            pass
    suite = unittest.TestSuite()
    suite.addTest(Reference1('test_something'))
    suite.addTest(Reference2('test_something'))
    tests = list(generate_scenarios(suite))
    self.assertEqual(4, len(tests))