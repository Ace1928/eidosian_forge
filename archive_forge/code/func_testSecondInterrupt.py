import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testSecondInterrupt(self):
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        self.skipTest('test requires SIGINT to not be ignored')

    def test(result):
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)
        result.breakCaught = True
        self.assertTrue(result.shouldStop)
        os.kill(pid, signal.SIGINT)
        self.fail('Second KeyboardInterrupt not raised')

    def test_function():
        result = unittest.TestResult()
        unittest.installHandler()
        unittest.registerResult(result)
        with self.assertRaises(KeyboardInterrupt):
            test(result)
        self.assertTrue(result.breakCaught)
    self.withRepeats(test_function)