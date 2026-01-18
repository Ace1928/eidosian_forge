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
def logInAs(self, nickname, password):
    d = self.factory.portal.login(credentials.UsernamePassword(nickname, password), self, iwords.IUser)
    d.addCallbacks(self._cbLogin, self._ebLogin, errbackArgs=(nickname,))