import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testRemoveHandlerAsDecorator(self):
    default_handler = signal.getsignal(signal.SIGINT)
    unittest.installHandler()

    @unittest.removeHandler
    def test():
        self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
    test()
    self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)