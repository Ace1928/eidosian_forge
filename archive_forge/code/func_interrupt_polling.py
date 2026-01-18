import sys
import time
from functools import wraps
from pytest import mark
from zmq.tests import BaseZMQTestCase
from zmq.utils.win32 import allow_interrupt
@count_calls
def interrupt_polling():
    print('Caught CTRL-C!')