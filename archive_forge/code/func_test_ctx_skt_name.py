import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_skt_name():

    @context('ctx')
    @socket('skt', zmq.PUB, context_name='ctx')
    def test(ctx, skt):
        assert isinstance(skt, zmq.Socket), skt
        assert isinstance(ctx, zmq.Context), ctx
        assert skt.type == zmq.PUB
    test()