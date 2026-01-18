import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_configure_add_with_multi_datadirs(self):
    """
        Tests multiple filesystem specified by filesystem_store_datadirs
        are parsed correctly.
        """
    store_map = [self.useFixture(fixtures.TempDir()).path, self.useFixture(fixtures.TempDir()).path]
    self.conf.set_override('filesystem_store_datadir', override=None, group='glance_store')
    self.conf.set_override('filesystem_store_datadirs', [store_map[0] + ':100', store_map[1] + ':200'], group='glance_store')
    self.store.configure_add()
    expected_priority_map = {100: [store_map[0]], 200: [store_map[1]]}
    expected_priority_list = [200, 100]
    self.assertEqual(expected_priority_map, self.store.priority_data_map)
    self.assertEqual(expected_priority_list, self.store.priority_list)