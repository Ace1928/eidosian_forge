import pytest
def test_ioloop():
    from zmq.eventloop import ioloop
    assert ioloop.IOLoop is tornado.ioloop.IOLoop
    assert ioloop.ZMQIOLoop is ioloop.IOLoop