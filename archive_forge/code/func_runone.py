import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def runone():
    sections.append(2)
    d = Deferred()
    reactor.callLater(1, d.callback, None)
    yield from d
    sections.append(3)
    return 'Yay!'