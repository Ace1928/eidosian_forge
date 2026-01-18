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
@mock.patch.object(vm_store._Reader, 'size')
@mock.patch('oslo_vmware.api.VMwareAPISession')
def test_add_size_zero(self, mock_api_session, fake_size, fake_select_datastore):
    """
        Test that when specifying size zero for the image to add,
        the actual size of the image is returned.
        """
    fake_select_datastore.return_value = self.store.datastores[0][0]
    expected_image_id = str(uuid.uuid4())
    expected_size = FIVE_KB
    expected_contents = b'*' * expected_size
    hash_code = secretutils.md5(expected_contents, usedforsecurity=False)
    expected_checksum = hash_code.hexdigest()
    sha256_code = hashlib.sha256(expected_contents)
    expected_multihash = sha256_code.hexdigest()
    fake_size.__get__ = mock.Mock(return_value=expected_size)
    with mock.patch('hashlib.md5') as md5:
        with mock.patch('hashlib.new') as fake_new:
            md5.return_value = hash_code
            fake_new.return_value = sha256_code
            expected_location = format_location(VMWARE_DS['vmware_server_host'], VMWARE_DS['vmware_store_image_dir'], expected_image_id, VMWARE_DS['vmware_datastores'])
            image = io.BytesIO(expected_contents)
            with mock.patch('requests.Session.request') as HttpConn:
                HttpConn.return_value = utils.fake_response()
                location, size, checksum, multihash, _ = self.store.add(expected_image_id, image, 0, self.hash_algo)
    self.assertEqual(utils.sort_url_by_qs_keys(expected_location), utils.sort_url_by_qs_keys(location))
    self.assertEqual(expected_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(expected_multihash, multihash)