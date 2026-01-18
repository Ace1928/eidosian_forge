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
def test_configure_add_check_metadata_id_or_mountpoint_is_not_string(self):
    metadata = {'id': 10, 'mountpoint': '/tmp'}
    self._create_metadata_json_file(metadata)
    self.assertRaises(exceptions.BadStoreConfiguration, self.store.configure_add)
    metadata = {'id': 'asdf1234', 'mountpoint': 12345}
    self._create_metadata_json_file(metadata)
    self.assertRaises(exceptions.BadStoreConfiguration, self.store.configure_add)