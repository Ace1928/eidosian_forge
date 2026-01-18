import unittest as pyunit
from twisted.trial import itrial, reporter, runner, unittest
from twisted.trial.test import mockdoctest
def test_basicTrialIntegration(self) -> None:
    """
        L{loadDoctests} loads all of the doctests in the given module.
        """
    loader = runner.TestLoader()
    suite = loader.loadDoctests(mockdoctest)
    self.assertEqual(7, suite.countTestCases())