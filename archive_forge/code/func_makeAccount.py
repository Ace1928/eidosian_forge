from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
def makeAccount(self):
    return DummyAccount('la', False, 'la', None, 'localhost', 6667)