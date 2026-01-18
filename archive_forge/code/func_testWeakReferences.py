import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testWeakReferences(self):
    result = unittest.TestResult()
    unittest.registerResult(result)
    ref = weakref.ref(result)
    del result
    gc.collect()
    gc.collect()
    self.assertIsNone(ref())