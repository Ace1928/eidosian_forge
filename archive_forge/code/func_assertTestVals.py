import functools
from unittest import mock
import uuid
from keystoneauth1 import loading
from keystoneauth1.loading import base
from keystoneauth1 import plugin
from keystoneauth1.tests.unit import utils
def assertTestVals(self, plugin, vals=TEST_VALS):
    for k, v in vals.items():
        self.assertEqual(v, plugin[k])