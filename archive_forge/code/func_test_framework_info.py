import os
import sys
import unittest
from ctypes.macholib.dyld import dyld_find
from ctypes.macholib.dylib import dylib_info
from ctypes.macholib.framework import framework_info
@unittest.skipUnless(sys.platform == 'darwin', 'OSX-specific test')
def test_framework_info(self):
    self.assertIsNone(framework_info('completely/invalid'))
    self.assertIsNone(framework_info('completely/invalid/_debug'))
    self.assertIsNone(framework_info('P/F.framework'))
    self.assertIsNone(framework_info('P/F.framework/_debug'))
    self.assertEqual(framework_info('P/F.framework/F'), d('P', 'F.framework/F', 'F'))
    self.assertEqual(framework_info('P/F.framework/F_debug'), d('P', 'F.framework/F_debug', 'F', suffix='debug'))
    self.assertIsNone(framework_info('P/F.framework/Versions'))
    self.assertIsNone(framework_info('P/F.framework/Versions/A'))
    self.assertEqual(framework_info('P/F.framework/Versions/A/F'), d('P', 'F.framework/Versions/A/F', 'F', 'A'))
    self.assertEqual(framework_info('P/F.framework/Versions/A/F_debug'), d('P', 'F.framework/Versions/A/F_debug', 'F', 'A', 'debug'))