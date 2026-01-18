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
def jellyFor(self, jellier):
    qual = reflect.qual(self.__class__)
    if isinstance(qual, str):
        qual = qual.encode('utf-8')
    return (qual, jellier.invoker.registerReference(self))