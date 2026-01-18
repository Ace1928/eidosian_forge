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
def test_get_datacenter_ref(self, mock_api_session):
    datacenter_path = 'Datacenter1'
    self.store._get_datacenter(datacenter_path)
    self.store.session.invoke_api.assert_called_with(self.store.session.vim, 'FindByInventoryPath', self.store.session.vim.service_content.searchIndex, inventoryPath=datacenter_path)