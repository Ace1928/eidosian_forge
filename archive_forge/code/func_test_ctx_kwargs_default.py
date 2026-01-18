import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_kwargs_default():

    @context(name='ctx', io_threads=5)
    def test(ctx=None):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS
    test()