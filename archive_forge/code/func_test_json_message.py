import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_json_message(self):
    payload = 'body: {"changePassword": {"adminPass": "1234567"}}'
    expected = 'body: {"changePassword": {"adminPass": "***"}}'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'body: {"rescue": {"admin_pass": "1234567"}}'
    expected = 'body: {"rescue": {"admin_pass": "***"}}'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'body: {"rescue": {"admin_password": "1234567"}}'
    expected = 'body: {"rescue": {"admin_password": "***"}}'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'body: {"rescue": {"password": "1234567"}}'
    expected = 'body: {"rescue": {"password": "***"}}'
    self.assertEqual(expected, strutils.mask_password(payload))