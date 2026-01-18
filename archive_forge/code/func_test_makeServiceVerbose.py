from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component
def test_makeServiceVerbose(self) -> None:
    """
        The verbose flag enables traffic logging.
        """
    opt = tap.Options()
    opt.parseOptions(['--verbose'])
    s = tap.makeService(opt)
    self.assertTrue(s.factory.logTraffic)