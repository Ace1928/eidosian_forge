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
def test_config_args(self):
    TestConfigurable.configure(None, a=3)
    obj = cast(TestConfig1, TestConfigurable())
    self.assertIsInstance(obj, TestConfig1)
    self.assertEqual(obj.a, 3)
    obj = cast(TestConfig1, TestConfigurable(42, a=4))
    self.assertIsInstance(obj, TestConfig1)
    self.assertEqual(obj.a, 4)
    self.assertEqual(obj.pos_arg, 42)
    self.checkSubclasses()
    obj = TestConfig1()
    self.assertIs(obj.a, None)