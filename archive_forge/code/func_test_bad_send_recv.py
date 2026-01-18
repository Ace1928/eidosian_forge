import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent
def test_bad_send_recv(self):
    s1, s2 = self.create_bound_pair(zmq.REQ, zmq.REP)
    if zmq.zmq_version() != '2.1.8':
        for copy in (True, False):
            self.assertRaisesErrno(zmq.EFSM, s1.recv, copy=copy)
            self.assertRaisesErrno(zmq.EFSM, s2.send, b'asdf', copy=copy)
    msg1 = b'asdf'
    msg2 = self.ping_pong(s1, s2, msg1)
    assert msg1 == msg2