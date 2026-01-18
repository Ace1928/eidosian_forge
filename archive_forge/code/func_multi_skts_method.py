import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
@context()
@socket(zmq.PUB)
@socket(zmq.SUB)
def multi_skts_method(self, ctx, pub, sub, foo='bar'):
    assert isinstance(self, TestMethodDecorators), self
    assert isinstance(ctx, zmq.Context), ctx
    assert isinstance(pub, zmq.Socket), pub
    assert isinstance(sub, zmq.Socket), sub
    assert foo == 'bar'
    assert pub.context is ctx
    assert sub.context is ctx
    assert pub.type == zmq.PUB
    assert sub.type == zmq.SUB