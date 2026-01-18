import hashlib
import io
from unittest import mock
import uuid
from oslo_utils import secretutils
from oslo_utils import units
from oslo_vmware import api
from oslo_vmware import exceptions as vmware_exceptions
from oslo_vmware.objects import datacenter as oslo_datacenter
from oslo_vmware.objects import datastore as oslo_datastore
import glance_store._drivers.vmware_datastore as vm_store
from glance_store import backend
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
@mock.patch('oslo_vmware.api.VMwareAPISession')
def test_get_size_non_existing(self, mock_api_session):
    """
        Test that trying to retrieve an image size that doesn't exist
        raises an error
        """
    loc = location.get_location_from_uri('vsphere://127.0.0.1/folder/openstack_glance/%s?dsName=ds1&dcPath=dc1' % FAKE_UUID, conf=self.conf)
    with mock.patch('requests.Session.request') as HttpConn:
        HttpConn.return_value = utils.fake_response(status_code=404)
        self.assertRaises(exceptions.NotFound, self.store.get_size, loc)