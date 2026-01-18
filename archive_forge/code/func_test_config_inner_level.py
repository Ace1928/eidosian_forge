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
def test_config_inner_level(self):
    obj = TestConfig3()
    self.assertIsInstance(obj, TestConfig3A)
    TestConfig3.configure(TestConfig3B)
    obj = TestConfig3()
    self.assertIsInstance(obj, TestConfig3B)
    obj2 = TestConfigurable()
    self.assertIsInstance(obj2, TestConfig1)
    TestConfigurable.configure(TestConfig2)
    obj3 = TestConfigurable()
    self.assertIsInstance(obj3, TestConfig2)
    obj = TestConfig3()
    self.assertIsInstance(obj, TestConfig3B)