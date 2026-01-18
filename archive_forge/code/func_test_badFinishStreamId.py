from random import Random
from typing import Awaitable, Dict, List, TypeVar, Union
from hamcrest import (
from hypothesis import given
from hypothesis.strategies import binary, integers, just, lists, randoms, text
from twisted.internet.defer import Deferred, fail
from twisted.internet.interfaces import IProtocol
from twisted.internet.protocol import Protocol
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, connect
from twisted.trial.unittest import SynchronousTestCase
from ..stream import StreamOpen, StreamReceiver, StreamWrite, chunk, stream
from .matchers import HasSum, IsSequenceOf
@given(integers())
def test_badFinishStreamId(self, streamId: int) -> None:
    """
        L{StreamReceiver.finish} raises L{KeyError} if called with a
        streamId not associated with an open stream.
        """
    receiver = StreamReceiver()
    assert_that(calling(receiver.finish).with_args(streamId), raises(KeyError))