import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_add_consumer_with_valid_parameters_doesnt_fail(self):
    self._mock_list_versions()
    self.key_mgr.add_consumer(self.ctxt, self.secret_ref, self._get_custom_consumer_data())