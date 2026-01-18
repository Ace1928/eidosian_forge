from unittest import mock
from castellan.tests.unit.key_manager import fake
from os_brick import encryptors
from os_brick.tests import base
def test_get_error_encryptors(self):
    encryption = {'control_location': 'front-end', 'provider': 'ErrorEncryptor'}
    self.assertRaises(ValueError, encryptors.get_volume_encryptor, root_helper=self.root_helper, connection_info=self.connection_info, keymgr=self.keymgr, **encryption)