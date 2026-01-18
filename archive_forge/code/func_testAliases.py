from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def testAliases(self):
    """
        Test that adding an aliases(5) file to an IAliasableDomain at least
        doesn't raise an unhandled exception.
        """
    Options().parseOptions(['--maildirdbmdomain', 'example.com=example.com', '--aliases', self.aliasFilename])