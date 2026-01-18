from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_datacenter_path(self):
    dc_path = 'datacenter-1'
    ds_name = 'datastore-1'
    params = {'dcPath': dc_path, 'dsName': ds_name}
    query = urlparse.urlencode(params)
    url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
    ds_url = datastore.DatastoreURL.urlparse(url)
    self.assertEqual(dc_path, ds_url.datacenter_path)