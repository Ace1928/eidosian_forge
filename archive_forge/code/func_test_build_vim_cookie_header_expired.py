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
@mock.patch.object(api, 'VMwareAPISession')
def test_build_vim_cookie_header_expired(self, mock_api_session):
    self.store.session.is_current_session_active = mock.Mock()
    self.store.session.is_current_session_active.return_value = False
    self.store._build_vim_cookie_header(True)
    self.assertTrue(mock_api_session.called)