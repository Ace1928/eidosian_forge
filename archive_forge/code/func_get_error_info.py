import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def get_error_info(self):
    """Return a text representation of an exception thrown by a test
        method.
        """
    if not self.err:
        return ''
    return self.test_result._exc_info_to_string(self.err, self.test_method)