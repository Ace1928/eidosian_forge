import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def hook_apply_scenarios(self):
    self.addCleanup(setattr, testscenarios.scenarios, 'apply_scenarios', apply_scenarios)
    log = []

    def capture(scenarios, test):
        log.append((scenarios, test))
        return apply_scenarios(scenarios, test)
    testscenarios.scenarios.apply_scenarios = capture
    return log