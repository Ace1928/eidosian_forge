import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testTwoResults(self):

    def test_function():
        unittest.installHandler()
        result = unittest.TestResult()
        unittest.registerResult(result)
        new_handler = signal.getsignal(signal.SIGINT)
        result2 = unittest.TestResult()
        unittest.registerResult(result2)
        self.assertEqual(signal.getsignal(signal.SIGINT), new_handler)
        result3 = unittest.TestResult()
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            self.fail('KeyboardInterrupt not handled')
        self.assertTrue(result.shouldStop)
        self.assertTrue(result2.shouldStop)
        self.assertFalse(result3.shouldStop)
    self.withRepeats(test_function)