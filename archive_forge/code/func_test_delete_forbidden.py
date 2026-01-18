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
def test_delete_forbidden(self):
    """
        Tests that trying to delete a file without permissions
        raises the correct error
        """
    image_id = str(uuid.uuid4())
    file_size = 5 * units.Ki
    file_contents = b'*' * file_size
    image_file = io.BytesIO(file_contents)
    loc, size, checksum, multihash, _ = self.store.add(image_id, image_file, file_size, self.hash_algo)
    uri = 'file:///%s/%s' % (self.test_dir, image_id)
    loc = location.get_location_from_uri(uri, conf=self.conf)
    with mock.patch.object(os, 'unlink') as unlink:
        e = OSError()
        e.errno = errno
        unlink.side_effect = e
        self.assertRaises(exceptions.Forbidden, self.store.delete, loc)
        self.store.get(loc)