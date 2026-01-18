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
@mock.patch('glance_store._drivers.vmware_datastore._Reader')
def test_add_with_verifier_size_zero(self, fake_reader, fake_select_ds):
    """Test that the verifier is passed to the _ChunkReader during add."""
    verifier = mock.MagicMock(name='mock_verifier')
    image_id = str(uuid.uuid4())
    size = FIVE_KB
    contents = b'*' * size
    image = io.BytesIO(contents)
    with mock.patch('requests.Session.request') as HttpConn:
        HttpConn.return_value = utils.fake_response()
        self.store.add(image_id, image, 0, self.hash_algo, verifier=verifier)
    fake_reader.assert_called_with(image, self.hash_algo, verifier)