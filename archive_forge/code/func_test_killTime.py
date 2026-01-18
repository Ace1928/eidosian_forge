from twisted.python.usage import UsageError
from twisted.runner import procmontap as tap
from twisted.runner.procmon import ProcessMonitor
from twisted.trial import unittest
def test_killTime(self) -> None:
    """
        The killtime option is recognised as a parameter and coerced to float.
        """
    opt = tap.Options()
    opt.parseOptions(['--killtime', '7.5', 'foo'])
    self.assertEqual(opt['killtime'], 7.5)