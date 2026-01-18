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
def test_get_key(self):
    original_secret_metadata = mock.Mock()
    original_secret_metadata.algorithm = mock.sentinel.alg
    original_secret_metadata.bit_length = mock.sentinel.bit
    original_secret_metadata.secret_type = 'symmetric'
    key_id = '43ed09c3-e551-4c24-b612-e619abe9b534'
    key_ref = 'http://localhost:9311/v1/secrets/' + key_id
    original_secret_metadata.secret_ref = key_ref
    created = timeutils.parse_isotime('2015-10-20 18:51:17+00:00')
    original_secret_metadata.created = created
    created_formatted = timeutils.parse_isotime(str(created))
    created_posix = calendar.timegm(created_formatted.timetuple())
    key_name = 'my key'
    original_secret_metadata.name = key_name
    original_secret_data = b'test key'
    original_secret_metadata.payload = original_secret_data
    self.mock_barbican.secrets.get.return_value = original_secret_metadata
    key = self.key_mgr.get(self.ctxt, self.key_id)
    self.get.assert_called_once_with(self.secret_ref)
    self.assertEqual(key_id, key.id)
    self.assertEqual(key_name, key.name)
    self.assertEqual(original_secret_data, key.get_encoded())
    self.assertEqual(created_posix, key.created)