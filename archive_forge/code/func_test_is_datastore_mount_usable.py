from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_is_datastore_mount_usable(self):
    m = MountInfo('readWrite', True, True)
    self.assertTrue(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('read', True, True)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('readWrite', False, True)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('readWrite', True, False)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('readWrite', False, False)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('readWrite', None, None)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))
    m = MountInfo('readWrite', None, True)
    self.assertFalse(datastore.Datastore.is_datastore_mount_usable(m))