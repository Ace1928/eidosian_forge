import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_int_from_bool_as_string(self):
    self.assertEqual(1, strutils.int_from_bool_as_string(True))
    self.assertEqual(0, strutils.int_from_bool_as_string(False))