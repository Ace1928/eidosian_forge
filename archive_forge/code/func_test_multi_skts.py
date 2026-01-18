import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_multi_skts():

    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def test(pub, sub, push):
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push
        assert pub.context is zmq.Context.instance()
        assert sub.context is zmq.Context.instance()
        assert push.context is zmq.Context.instance()
        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH
    test()