from zope.interface import implementer
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import (
from twisted.cred.credentials import (
from twisted.cred.error import LoginDenied
from twisted.cred.portal import Portal
from twisted.internet.interfaces import IReactorProcess
from twisted.python.fakepwd import UserDatabase
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from .test_session import StubClient, StubConnection
def testExecCommandEnvironment(self) -> None:
    """
        C{execCommand} sets the C{HOME} environment variable to the avatar's home
        directory.
        """
    userdb = UserDatabase()
    homeDirectory = '/made/up/path/'
    userName = 'user'
    userdb.addUser(userName, home=homeDirectory)
    self.patch(unix, 'pwd', userdb)
    mockReactor = MockProcessSpawner()
    avatar = UnixConchUser(userName)
    avatar.conn = StubConnection(transport=StubClient())
    session = unix.SSHSessionForUnixConchUser(avatar, reactor=mockReactor)
    protocol = None
    command = ['not-actually-executed']
    session.execCommand(protocol, command)
    [call] = mockReactor._spawnProcessCalls
    self.assertEqual(homeDirectory, call['env']['HOME'])