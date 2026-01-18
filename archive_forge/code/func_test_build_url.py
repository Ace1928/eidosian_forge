from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_url(self):
    ds = datastore.Datastore('fake_ref', 'ds_name')
    path = 'images/ubuntu.vmdk'
    self.assertRaises(ValueError, ds.build_url, 'https', '10.0.0.2', path)
    ds.datacenter = mock.Mock()
    ds.datacenter.name = 'dc_path'
    ds_url = ds.build_url('https', '10.0.0.2', path)
    self.assertEqual(ds_url.datastore_name, 'ds_name')
    self.assertEqual(ds_url.datacenter_path, 'dc_path')
    self.assertEqual(ds_url.path, path)