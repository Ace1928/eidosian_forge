import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from os_brick.encryptors import cryptsetup
from os_brick import exception
from os_brick.tests.encryptors import test_base
@mock.patch('os_brick.executor.Executor._execute')
@mock.patch('os.path.exists', side_effect=[False, True])
def test_init_volume_encryption_with_wwn(self, mock_exists, mock_execute):
    old_dev_name = self.dev_path.split('/')[-1]
    wwn = 'fake_wwn'
    connection_info = copy.deepcopy(self.connection_info)
    connection_info['data']['multipath_id'] = wwn
    encryptor = cryptsetup.CryptsetupEncryptor(root_helper=self.root_helper, connection_info=connection_info, keymgr=fake.fake_api(), execute=mock_execute)
    self.assertFalse(encryptor.dev_name.startswith('crypt-'))
    self.assertEqual(wwn, encryptor.dev_name)
    self.assertEqual(self.dev_path, encryptor.dev_path)
    self.assertEqual(self.symlink_path, encryptor.symlink_path)
    mock_exists.assert_has_calls([mock.call('/dev/mapper/%s' % old_dev_name), mock.call('/dev/mapper/%s' % wwn)])
    mock_execute.assert_called_once_with('cryptsetup', 'status', wwn, run_as_root=True)