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
def test_add_check_metadata_list_with_invalid_mountpoint_locations(self):
    in_metadata = [{'id': 'abcdefg', 'mountpoint': '/xyz/images'}, {'id': 'xyz1234', 'mountpoint': '/pqr/images'}]
    location, size, checksum, multihash, metadata = self._store_image(in_metadata)
    self.assertEqual({}, metadata)