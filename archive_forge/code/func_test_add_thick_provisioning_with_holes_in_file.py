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
def test_add_thick_provisioning_with_holes_in_file(self):
    """
        Tests that a file which contains null bytes chunks is fully
        written with a thick provisioning configuration.
        """
    chunk_size = units.Ki
    content = b'*' * chunk_size + b'\x00' * chunk_size + b'*' * chunk_size
    self._do_test_thin_provisioning(content, 3 * chunk_size, 0, 3, False)