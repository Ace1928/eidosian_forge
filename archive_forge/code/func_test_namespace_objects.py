import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_namespace_objects(self):
    payload = '\n        Namespace(passcode=\'\', username=\'\', password=\'my"password\',\n        profile=\'\', verify=None, token=\'\')\n        '
    expected = "\n        Namespace(passcode='', username='', password='***',\n        profile='', verify=None, token='***')\n        "
    self.assertEqual(expected, strutils.mask_password(payload))