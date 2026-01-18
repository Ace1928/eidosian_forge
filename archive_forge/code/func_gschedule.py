from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def gschedule():
    reactor.callLater(0, callback)
    return 0