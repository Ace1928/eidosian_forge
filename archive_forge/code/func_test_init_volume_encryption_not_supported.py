import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from os_brick.encryptors import cryptsetup
from os_brick import exception
from os_brick.tests.encryptors import test_base
def test_init_volume_encryption_not_supported(self):
    type = 'unencryptable'
    data = dict(volume_id='a194699b-aa07-4433-a945-a5d23802043e')
    connection_info = dict(driver_volume_type=type, data=data)
    exc = self.assertRaises(exception.VolumeEncryptionNotSupported, cryptsetup.CryptsetupEncryptor, root_helper=self.root_helper, connection_info=connection_info, keymgr=fake.fake_api())
    self.assertIn(type, str(exc))