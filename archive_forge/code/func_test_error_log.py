from unittest import mock
from castellan.tests.unit.key_manager import fake
from os_brick import encryptors
from os_brick.tests import base
@mock.patch('os_brick.encryptors.LOG')
def test_error_log(self, log):
    encryption = {'control_location': 'front-end', 'provider': 'TestEncryptor'}
    provider = 'TestEncryptor'
    try:
        encryptors.get_volume_encryptor(root_helper=self.root_helper, connection_info=self.connection_info, keymgr=self.keymgr, **encryption)
    except Exception as e:
        log.error.assert_called_once_with('Error instantiating %(provider)s: %(exception)s', {'provider': provider, 'exception': e})