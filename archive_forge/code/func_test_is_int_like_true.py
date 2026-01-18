import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_is_int_like_true(self):
    self.assertTrue(strutils.is_int_like(1))
    self.assertTrue(strutils.is_int_like('1'))
    self.assertTrue(strutils.is_int_like('514'))
    self.assertTrue(strutils.is_int_like('0'))