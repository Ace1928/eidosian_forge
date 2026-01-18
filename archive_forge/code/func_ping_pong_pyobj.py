import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase
from pytest import mark
import zmq
from zmq.utils import jsonapi
def ping_pong_pyobj(self, s1, s2, o):
    s1.send_pyobj(o)
    o2 = s2.recv_pyobj()
    s2.send_pyobj(o2)
    o3 = s1.recv_pyobj()
    return o3