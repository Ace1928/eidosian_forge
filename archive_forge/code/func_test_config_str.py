from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
def test_config_str(self):
    TestConfigurable.configure('tornado.test.util_test.TestConfig2')
    obj = cast(TestConfig2, TestConfigurable())
    self.assertIsInstance(obj, TestConfig2)
    self.assertIs(obj.b, None)
    obj = cast(TestConfig2, TestConfigurable(b=2))
    self.assertIsInstance(obj, TestConfig2)
    self.assertEqual(obj.b, 2)
    self.checkSubclasses()