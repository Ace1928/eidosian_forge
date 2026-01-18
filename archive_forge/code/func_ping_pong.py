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
def ping_pong(self, s1, s2, msg):
    s1.send(msg)
    msg2 = s2.recv()
    s2.send(msg2)
    msg3 = s1.recv()
    return msg3