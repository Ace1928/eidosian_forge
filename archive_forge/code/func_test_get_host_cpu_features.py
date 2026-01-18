import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_get_host_cpu_features(self):
    features = llvm.get_host_cpu_features()
    self.assertIsInstance(features, dict)
    self.assertIsInstance(features, llvm.FeatureMap)
    for k, v in features.items():
        self.assertIsInstance(k, str)
        self.assertTrue(k)
        self.assertIsInstance(v, bool)
    self.assertIsInstance(features.flatten(), str)
    re_term = '[+\\-][a-zA-Z0-9\\._-]+'
    regex = '^({0}|{0}(,{0})*)?$'.format(re_term)
    self.assertIsNotNone(re.match(regex, ''))
    self.assertIsNotNone(re.match(regex, '+aa'))
    self.assertIsNotNone(re.match(regex, '+a,-bb'))
    if len(features) == 0:
        self.assertEqual(features.flatten(), '')
    else:
        self.assertIsNotNone(re.match(regex, features.flatten()))