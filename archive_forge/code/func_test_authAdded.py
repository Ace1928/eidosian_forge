from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_authAdded(self) -> None:
    """
        The C{--auth} command-line option will add a checker to the list of
        checkers, and it should be the only auth checker
        """
    self.options.parseOptions(['--auth', 'file:' + self.filename])
    self.assertEqual(len(self.options['credCheckers']), 1)