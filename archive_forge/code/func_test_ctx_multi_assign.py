import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_multi_assign():

    @context(name='ctx')
    def test(ctx):
        pass
    with raises(TypeError):
        test('mock')