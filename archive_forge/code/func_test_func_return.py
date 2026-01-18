import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_func_return():

    @context()
    def f(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        return 'something'
    assert f() == 'something'