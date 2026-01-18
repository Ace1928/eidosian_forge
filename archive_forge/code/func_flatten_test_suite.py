import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
def flatten_test_suite(test_suite, ret):
    if isinstance(test_suite, unittest.TestSuite):
        for t in test_suite._tests:
            flatten_test_suite(t, ret)
    elif isinstance(test_suite, unittest.TestCase):
        ret.append(test_suite)