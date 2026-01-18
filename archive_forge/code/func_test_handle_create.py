from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_handle_create(self):
    value = mock.MagicMock()
    volume_type_id = '01bd581d-33fe-4d6d-bd7b-70ae076d39fb'
    value.volume_type_id = volume_type_id
    self.volume_encryption_types.create.return_value = value
    with mock.patch.object(self.my_encrypted_vol_type.client_plugin(), 'get_volume_type') as mock_get_volume_type:
        mock_get_volume_type.return_value = volume_type_id
        self.my_encrypted_vol_type.handle_create()
        mock_get_volume_type.assert_called_once_with(volume_type_id)
    specs = {'control_location': 'front-end', 'cipher': 'aes-xts-plain64', 'key_size': 512, 'provider': 'nova.volume.encryptors.luks.LuksEncryptor'}
    self.volume_encryption_types.create.assert_called_once_with(volume_type=volume_type_id, specs=specs)
    self.assertEqual(volume_type_id, self.my_encrypted_vol_type.resource_id)