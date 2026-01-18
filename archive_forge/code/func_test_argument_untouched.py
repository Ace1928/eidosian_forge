import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_argument_untouched(self):
    """Make sure that the argument passed in is not modified"""
    payload = {'password': 'DK0PK1AK3', 'bool': True, 'dict': {'cat': 'meow', 'password': '*aa38skdjf'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
    pristine = copy.deepcopy(payload)
    strutils.mask_dict_password(payload)
    self.assertEqual(pristine, payload)