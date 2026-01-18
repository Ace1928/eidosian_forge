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
def test_delete_secret_with_consumers_force_parameter_false(self):
    self.mock_barbican.secrets.delete = mock.Mock(side_effect=barbican_exceptions.HTTPClientError("Secret has consumers! Use the 'force' parameter."))
    self.assertRaises(exception.KeyManagerError, self.key_mgr.delete, self.ctxt, self.key_id, force=False)
    self.mock_barbican.secrets.delete.assert_called_once_with(self.secret_ref, False)