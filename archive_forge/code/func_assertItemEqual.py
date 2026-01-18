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
def assertItemEqual(self, expected, actual):
    """Compare item considering formattable columns.

        This method compares an observed item to an expected item column by
        column. If a column is a formattable column, observed and expected
        columns are compared using human_readable() and machine_readable().
        """
    self.assertEqual(len(expected), len(actual))
    for col_expected, col_actual in zip(expected, actual):
        if isinstance(col_expected, cliff_columns.FormattableColumn):
            self.assertIsInstance(col_actual, col_expected.__class__)
            self.assertEqual(col_expected.human_readable(), col_actual.human_readable())
            self.assertEqual(col_expected.machine_readable(), col_actual.machine_readable())
        else:
            self.assertEqual(col_expected, col_actual)