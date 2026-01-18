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
@mock.patch.object(vm_store.Store, 'select_datastore')
@mock.patch.object(api, 'VMwareAPISession')
def test_add_ioerror(self, mock_api_session, mock_select_datastore):
    mock_select_datastore.return_value = self.store.datastores[0][0]
    expected_image_id = str(uuid.uuid4())
    expected_size = FIVE_KB
    expected_contents = b'*' * expected_size
    image = io.BytesIO(expected_contents)
    self.session = mock.Mock()
    with mock.patch('requests.Session.request') as HttpConn:
        HttpConn.request.side_effect = IOError
        self.assertRaises(exceptions.BackendException, self.store.add, expected_image_id, image, expected_size, self.hash_algo)