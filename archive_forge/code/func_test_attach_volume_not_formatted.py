import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from oslo_concurrency import processutils as putils
from os_brick.encryptors import luks
from os_brick import exception
from os_brick.tests.encryptors import test_base
@mock.patch('os_brick.executor.Executor._execute')
def test_attach_volume_not_formatted(self, mock_execute):
    fake_key = 'bc37c5eccebe403f9cc2d0dd20dac2bc'
    self.encryptor._get_key = mock.MagicMock()
    self.encryptor._get_key.return_value = fake__get_key(None, fake_key)
    mock_execute.side_effect = [putils.ProcessExecutionError(exit_code=1), putils.ProcessExecutionError(exit_code=1), mock.DEFAULT, mock.DEFAULT, mock.DEFAULT]
    self.encryptor.attach_volume(None)
    mock_execute.assert_has_calls([mock.call('cryptsetup', 'luksOpen', '--key-file=-', self.dev_path, self.dev_name, process_input=fake_key, root_helper=self.root_helper, run_as_root=True, check_exit_code=True), mock.call('cryptsetup', 'isLuks', '--verbose', self.dev_path, root_helper=self.root_helper, run_as_root=True, check_exit_code=True), mock.call('cryptsetup', '--batch-mode', 'luksFormat', '--type', 'luks2', '--key-file=-', self.dev_path, process_input=fake_key, root_helper=self.root_helper, run_as_root=True, check_exit_code=True, attempts=3), mock.call('cryptsetup', 'luksOpen', '--key-file=-', self.dev_path, self.dev_name, process_input=fake_key, root_helper=self.root_helper, run_as_root=True, check_exit_code=True), mock.call('ln', '--symbolic', '--force', '/dev/mapper/%s' % self.dev_name, self.symlink_path, root_helper=self.root_helper, run_as_root=True, check_exit_code=True)], any_order=False)