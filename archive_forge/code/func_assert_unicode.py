import sys
import unittest
from unittest import mock
from bpython.curtsiesfrontend.coderunner import CodeRunner, FakeOutput
def assert_unicode(self, s):
    self.assertIsInstance(s, str)