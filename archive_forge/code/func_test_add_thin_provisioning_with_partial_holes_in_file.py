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
def test_add_thin_provisioning_with_partial_holes_in_file(self):
    """
        Tests that a file which contains null bytes not aligned with
        chunk size is sparsified with a thin provisioning configuration.
        """
    chunk_size = units.Ki
    my_chunk = int(chunk_size * 1.5)
    content = b'*' * my_chunk + b'\x00' * my_chunk + b'*' * my_chunk
    self._do_test_thin_provisioning(content, 3 * my_chunk, 1, 4, True)