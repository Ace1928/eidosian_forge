from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_defaultAuths(self) -> None:
    """
        Make sure that if the C{--auth} command-line option is not passed,
        the default checkers are (for backwards compatibility): SSH and UNIX
        """
    numCheckers = 2
    self.assertIn(ISSHPrivateKey, self.options['credInterfaces'], 'SSH should be one of the default checkers')
    self.assertIn(IUsernamePassword, self.options['credInterfaces'], 'UNIX should be one of the default checkers')
    self.assertEqual(numCheckers, len(self.options['credCheckers']), 'There should be %d checkers by default' % (numCheckers,))