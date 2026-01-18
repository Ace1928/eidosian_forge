from pytest import mark
import zmq
@only_bundled
def test_has_ipc():
    """bundled libzmq has ipc support"""
    assert zmq.has('ipc')