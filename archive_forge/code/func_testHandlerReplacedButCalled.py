import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testHandlerReplacedButCalled(self):
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        self.skipTest('test requires SIGINT to not be ignored')

    def test_function():
        unittest.installHandler()
        handler = signal.getsignal(signal.SIGINT)

        def new_handler(frame, signum):
            handler(frame, signum)
        signal.signal(signal.SIGINT, new_handler)
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            pass
        else:
            self.fail("replaced but delegated handler doesn't raise interrupt")
    self.withRepeats(test_function)