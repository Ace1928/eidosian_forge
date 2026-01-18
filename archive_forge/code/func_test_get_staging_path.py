import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_staging_path(self):
    expected = os.path.join(self.test_dir, 'staging')
    self.assertEqual(expected, housekeeping.staging_store_path())