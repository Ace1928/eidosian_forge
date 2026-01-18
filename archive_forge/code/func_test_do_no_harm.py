import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_do_no_harm(self):
    payload = {}
    expected = {}
    self.assertEqual(expected, strutils.mask_dict_password(payload))
    payload = {'somekey': 'somevalue', 'anotherkey': 'anothervalue'}
    expected = {'somekey': 'somevalue', 'anotherkey': 'anothervalue'}
    self.assertEqual(expected, strutils.mask_dict_password(payload))