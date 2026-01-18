import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent
def test_device_bind_to_random_binderror(self):
    dev = devices.ThreadDevice(zmq.PULL, zmq.PUSH, -1)
    iface = 'tcp://127.0.0.1'
    try:
        for i in range(11):
            dev.bind_in_to_random_port(iface, min_port=10000, max_port=10010)
    except zmq.ZMQBindError as e:
        return
    else:
        self.fail('Should have failed')