from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
def test_node_reference_create_duplicate(self):
    with mock.patch('glance.db.get_api') as mock_get_db:
        self.db = unit_test_utils.FakeDB(initialize=False)
        mock_get_db.return_value = self.db
        with mock.patch.object(self.db, 'node_reference_create') as mock_node_create:
            mock_node_create.side_effect = exception.Duplicate
            with mock.patch.object(image_cache.drivers.centralized_db, 'LOG') as mock_log:
                image_cache.ImageCache()
                expected_calls = [mock.call('Node reference is already recorded, ignoring it')]
                mock_log.debug.assert_has_calls(expected_calls)