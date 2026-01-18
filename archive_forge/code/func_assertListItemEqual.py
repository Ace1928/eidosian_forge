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
def assertListItemEqual(self, expected, actual):
    """Compare a list of items considering formattable columns.

        Each pair of observed and expected items are compared
        using assertItemEqual() method.
        """
    self.assertEqual(len(expected), len(actual))
    for item_expected, item_actual in zip(expected, actual):
        self.assertItemEqual(item_expected, item_actual)