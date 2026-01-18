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
def cbGroups(groups):

    def gotSize(size, group):
        return (group.name, size, group.meta.get('topic'))
    d = defer.DeferredList([group.size().addCallback(gotSize, group) for group in groups])
    d.addCallback(lambda results: self.list([r for s, r in results if s]))
    return d