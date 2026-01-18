import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def makeMethod(self, fname, args):

    def method(*a, **kw):
        if len(a) > len(args):
            raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
        for name, value in zip(args, a):
            if name in kw:
                raise TypeError("TypeError: %s() got multiple values for keyword argument '%s'" % (fname, name))
            else:
                kw[name] = value
        if len(kw) != len(args):
            raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
        self.calls.append((fname, kw))
    return method