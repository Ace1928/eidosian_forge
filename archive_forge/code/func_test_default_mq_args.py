import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def test_default_mq_args(self):
    self.device = dev = devices.ThreadMonitoredQueue(zmq.ROUTER, zmq.DEALER, zmq.PUB)
    dev.setsockopt_in(zmq.LINGER, 0)
    dev.setsockopt_out(zmq.LINGER, 0)
    dev.setsockopt_mon(zmq.LINGER, 0)
    dev.start()
    self.teardown_device()