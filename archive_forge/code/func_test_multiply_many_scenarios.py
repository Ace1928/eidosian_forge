import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_multiply_many_scenarios(self):

    def factory(name):
        for i in 'abc':
            yield (i, {name: i})
    scenarios = multiply_scenarios(factory('p'), factory('q'), factory('r'), factory('t'))
    self.assertEqual(3 ** 4, len(scenarios), scenarios)
    self.assertEqual('a,a,a,a', scenarios[0][0])