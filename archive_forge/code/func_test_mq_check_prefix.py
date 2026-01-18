import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def test_mq_check_prefix(self):
    ins = self.context.socket(zmq.ROUTER)
    outs = self.context.socket(zmq.DEALER)
    mons = self.context.socket(zmq.PUB)
    self.sockets.extend([ins, outs, mons])
    ins = 'in'
    outs = 'out'
    self.assertRaises(TypeError, devices.monitoredqueue, ins, outs, mons)