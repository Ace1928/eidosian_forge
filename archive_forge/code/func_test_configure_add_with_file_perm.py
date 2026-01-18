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
def test_configure_add_with_file_perm(self):
    """
        Tests filesystem specified by filesystem_store_file_perm
        are parsed correctly.
        """
    store = self.useFixture(fixtures.TempDir()).path
    self.conf.set_override('filesystem_store_datadir', store, group='glance_store')
    self.conf.set_override('filesystem_store_file_perm', 700, group='glance_store')
    self.store.configure_add()
    self.assertEqual(self.store.datadir, store)