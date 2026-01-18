import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_reinit():
    result = {'foo': None, 'bar': None}

    @context()
    def f(key, ctx):
        assert isinstance(ctx, zmq.Context), ctx
        result[key] = ctx
    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))
    foo_t.start()
    bar_t.start()
    foo_t.join()
    bar_t.join()
    assert result['foo'] is not None, result
    assert result['bar'] is not None, result
    assert result['foo'] is not result['bar'], result