import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_bool_bool_from_string(self):
    self.assertTrue(strutils.bool_from_string(True))
    self.assertFalse(strutils.bool_from_string(False))