import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_close_open(self):
    """Test closing and re-opening keeps state"""
    q = self.queue()
    q.push(b'a')
    q.push(b'b')
    q.push(b'c')
    q.push(b'd')
    q.pop()
    q.pop()
    q.close()
    del q
    q = self.queue()
    self.assertEqual(len(q), 2)
    q.push(b'e')
    q.pop()
    q.pop()
    q.close()
    del q
    q = self.queue()
    assert q.pop() is not None
    self.assertEqual(len(q), 0)