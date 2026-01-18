import pytest
def test_ioloop_install():
    from zmq.eventloop import ioloop
    with pytest.warns(DeprecationWarning):
        ioloop.install()