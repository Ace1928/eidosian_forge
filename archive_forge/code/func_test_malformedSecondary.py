from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase
def test_malformedSecondary(self) -> None:
    """
        If the value supplied for an I{--secondary} option does not provide a
        server IP address, optional port number, and domain name,
        L{Options.parseOptions} raises L{UsageError}.
        """
    options = Options()
    self.assertRaises(UsageError, options.parseOptions, ['--secondary', ''])
    self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4'])
    self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4:hello'])
    self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4:hello/example.com'])