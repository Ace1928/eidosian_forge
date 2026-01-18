from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def irc_NICKSERV_PRIVMSG(self, prefix, params):
    """
        Send a (private) message.

        Parameters: <msgtarget> <text to be sent>
        """
    target = params[0]
    password = params[-1]
    if self.nickname is None:
        self.transport.loseConnection()
    elif target.lower() != 'nickserv':
        self.privmsg(NICKSERV, self.nickname, 'Denied.  Please send me (NickServ) your password.')
    else:
        nickname = self.nickname
        self.nickname = None
        self.logInAs(nickname, password)