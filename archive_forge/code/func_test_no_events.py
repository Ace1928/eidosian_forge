import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
def test_no_events(self):
    s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    poller = self.Poller()
    poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
    poller.register(s2, 0)
    assert s1 in poller
    assert s2 not in poller
    poller.register(s1, 0)
    assert s1 not in poller