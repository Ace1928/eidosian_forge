from typing import Any, Tuple, Union
from twisted.application.internet import StreamServerEndpointService
from twisted.cred import error
from twisted.cred.checkers import FilePasswordDB, ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, UsernamePassword
from twisted.internet.defer import Deferred
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_authFailure(self) -> Any:
    """
        The checker created by the C{--auth} command-line option returns a
        L{Deferred} that fails with L{UnauthorizedLogin} when
        presented with credentials that are unknown to that checker.
        """
    self.options.parseOptions(['--auth', 'file:' + self.filename])
    checker: FilePasswordDB = self.options['credCheckers'][-1]
    self.assertIsInstance(checker, FilePasswordDB)
    invalid = UsernamePassword(self.usernamePassword[0], b'fake')
    return self.assertFailure(checker.requestAvatarId(invalid), error.UnauthorizedLogin)