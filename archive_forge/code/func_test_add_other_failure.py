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
def test_add_other_failure(self):
    """
        Tests that a non-space-related IOError does not raise a
        StorageFull exceptions.
        """
    self._do_test_add_write_failure(errno.ENOTDIR, IOError)