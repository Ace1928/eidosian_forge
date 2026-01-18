import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent
def test_large_msg(self):
    s1, s2 = self.create_bound_pair(zmq.REQ, zmq.REP)
    msg1 = 10000 * b'X'
    for i in range(10):
        msg2 = self.ping_pong(s1, s2, msg1)
        assert msg1 == msg2