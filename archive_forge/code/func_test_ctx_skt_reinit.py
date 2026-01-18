import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_skt_reinit():
    result = {'foo': {'ctx': None, 'skt': None}, 'bar': {'ctx': None, 'skt': None}}

    @context()
    @socket(zmq.PUB)
    def f(key, ctx, skt):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(skt, zmq.Socket), skt
        result[key]['ctx'] = ctx
        result[key]['skt'] = skt
    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))
    foo_t.start()
    bar_t.start()
    foo_t.join()
    bar_t.join()
    assert result['foo']['ctx'] is not None, result
    assert result['foo']['skt'] is not None, result
    assert result['bar']['ctx'] is not None, result
    assert result['bar']['skt'] is not None, result
    assert result['foo']['ctx'] is not result['bar']['ctx'], result
    assert result['foo']['skt'] is not result['bar']['skt'], result