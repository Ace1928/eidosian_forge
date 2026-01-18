from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_path(self):
    ds = datastore.Datastore('fake_ref', 'ds_name')
    ds_path = ds.build_path('some_dir', 'foo.vmdk')
    self.assertEqual('[ds_name] some_dir/foo.vmdk', str(ds_path))