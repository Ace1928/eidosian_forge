from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_no_capacity_no_freespace(self):
    ds = datastore.Datastore('fake_ref', 'ds_name')
    self.assertIsNone(ds.capacity)
    self.assertIsNone(ds.freespace)