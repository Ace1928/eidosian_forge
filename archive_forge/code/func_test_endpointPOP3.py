from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_endpointPOP3(self):
    """
        When I{--pop3} is given a TCP endpoint description as an argument, a
        TCPServerEndpoint is added to the list of POP3 endpoints.
        """
    self._endpointTest('pop3')