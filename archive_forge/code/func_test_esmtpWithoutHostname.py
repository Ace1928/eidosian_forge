from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_esmtpWithoutHostname(self):
    """
        If I{--esmtp} is given without I{--hostname}, L{Options.parseOptions}
        raises L{UsageError}.
        """
    options = Options()
    exc = self.assertRaises(UsageError, options.parseOptions, ['--esmtp'])
    self.assertEqual('--esmtp requires --hostname', str(exc))