import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
def start_zap(self):
    self.zap_thread = Thread(target=self.zap_handler)
    self.zap_thread.start()