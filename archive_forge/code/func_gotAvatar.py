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
def gotAvatar(avatar):
    if avatar.realm is not None:
        raise ewords.AlreadyLoggedIn()
    for iface in interfaces:
        facet = iface(avatar, None)
        if facet is not None:
            avatar.loggedIn(self, mind)
            mind.name = avatarId
            mind.realm = self
            mind.avatar = avatar
            return (iface, facet, self.logoutFactory(avatar, facet))
    raise NotImplementedError(self, interfaces)