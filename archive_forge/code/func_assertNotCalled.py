import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
def assertNotCalled(self, m, msg=None):
    """Assert a function was not called"""
    if m.called:
        if not msg:
            msg = 'method %s should not have been called' % m
        self.fail(msg)