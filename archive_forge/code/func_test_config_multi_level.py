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
def test_config_multi_level(self):
    TestConfigurable.configure(TestConfig3, a=1)
    obj = cast(TestConfig3A, TestConfigurable())
    self.assertIsInstance(obj, TestConfig3A)
    self.assertEqual(obj.a, 1)
    TestConfigurable.configure(TestConfig3)
    TestConfig3.configure(TestConfig3B, b=2)
    obj2 = cast(TestConfig3B, TestConfigurable())
    self.assertIsInstance(obj2, TestConfig3B)
    self.assertEqual(obj2.b, 2)