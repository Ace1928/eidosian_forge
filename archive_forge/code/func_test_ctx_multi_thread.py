import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_multi_thread():

    @context()
    @context()
    def f(foo, bar):
        assert isinstance(foo, zmq.Context), foo
        assert isinstance(bar, zmq.Context), bar
        assert len(set(map(id, [foo, bar]))) == 2, set(map(id, [foo, bar]))
    threads = [threading.Thread(target=f) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]