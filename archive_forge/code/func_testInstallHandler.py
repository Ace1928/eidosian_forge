import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testInstallHandler(self):
    default_handler = signal.getsignal(signal.SIGINT)
    unittest.installHandler()
    self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
    try:
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)
    except KeyboardInterrupt:
        self.fail('KeyboardInterrupt not handled')
    self.assertTrue(unittest.signals._interrupt_handler.called)