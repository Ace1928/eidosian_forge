import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
def setupKeyChecker(self, portal, users):
    """
        Create an L{ISSHPrivateKey} checker which recognizes C{users} and add it
        to C{portal}.

        @param portal: A L{Portal} to which to add the checker.
        @type portal: L{Portal}

        @param users: The users and their keys the checker will recognize.  Keys
            are byte strings giving user names.  Values are byte strings giving
            OpenSSH-formatted private keys.
        @type users: L{dict}
        """
    mapping = {k: [Key.fromString(v).public()] for k, v in users.items()}
    checker = SSHPublicKeyChecker(InMemorySSHKeyDB(mapping))
    portal.registerChecker(checker)