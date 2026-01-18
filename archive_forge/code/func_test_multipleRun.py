import signal
import time
from types import FrameType
from typing import Callable, List, Optional, Tuple, Union, cast
from twisted.internet.abstract import FileDescriptor
from twisted.internet.defer import Deferred
from twisted.internet.error import ReactorAlreadyRunning, ReactorNotRestartable
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_multipleRun(self) -> None:
    """
        C{reactor.run()} raises L{ReactorAlreadyRunning} when called when
        the reactor is already running.
        """
    events: List[str] = []
    testCase = cast(SynchronousTestCase, self)

    def reentrantRun() -> None:
        testCase.assertRaises(ReactorAlreadyRunning, reactor.run)
        events.append('tested')
    reactor = self.buildReactor()
    reactor.callWhenRunning(reentrantRun)
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    testCase.assertEqual(events, ['tested'])