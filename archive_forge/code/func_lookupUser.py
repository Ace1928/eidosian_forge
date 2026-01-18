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
def lookupUser(self, name):
    name = name.lower()
    try:
        user = self.users[name]
    except KeyError:
        return defer.fail(failure.Failure(ewords.NoSuchUser(name)))
    else:
        return defer.succeed(user)