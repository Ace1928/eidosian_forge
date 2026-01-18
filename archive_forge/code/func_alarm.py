import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
def alarm(self, t=None):
    """start a timer to fire only once

        like signal.alarm, but with better resolution than integer seconds.
        """
    if not hasattr(signal, 'setitimer'):
        raise SkipTest('EINTR tests require setitimer')
    if t is None:
        t = self.signal_delay
    self.timer_fired = False
    self.orig_handler = signal.signal(signal.SIGALRM, self.stop_timer)
    signal.setitimer(signal.ITIMER_REAL, t, 1000)