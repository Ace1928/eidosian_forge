from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds(self):
    ds = datastore.Datastore('fake_ref', 'ds_name', 2 * units.Gi, 1 * units.Gi, 1 * units.Gi)
    self.assertEqual('ds_name', ds.name)
    self.assertEqual('fake_ref', ds.ref)
    self.assertEqual(2 * units.Gi, ds.capacity)
    self.assertEqual(1 * units.Gi, ds.freespace)
    self.assertEqual(1 * units.Gi, ds.uncommitted)