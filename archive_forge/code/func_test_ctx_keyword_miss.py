import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_keyword_miss():

    @context(name='ctx')
    def test(other_name):
        pass
    with raises(TypeError):
        test()