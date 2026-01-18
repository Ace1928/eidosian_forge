from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server
def soap_complex(self):
    return {'a': ['b', 'c', 12, []], 'D': 'foo'}