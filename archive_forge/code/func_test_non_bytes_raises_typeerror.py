import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
@pytest.mark.xfail(reason='Reenable once Scrapy.squeues stops extending from this testsuite')
def test_non_bytes_raises_typeerror(self):
    q = self.queue()
    self.assertRaises(TypeError, q.push, 0)
    self.assertRaises(TypeError, q.push, u'')
    self.assertRaises(TypeError, q.push, None)
    self.assertRaises(TypeError, q.push, lambda x: x)