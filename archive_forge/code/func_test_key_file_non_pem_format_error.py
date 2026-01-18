import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
@unittest.skipIf(paramiko_version >= (2, 7, 0), 'New versions of paramiko support OPENSSH key format')
def test_key_file_non_pem_format_error(self):
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa_non_pem_format.key')
    with open(path) as fp:
        private_key = fp.read()
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_material': private_key}
    mock = ParamikoSSHClient(**conn_params)
    expected_msg = 'Invalid or unsupported key type'
    assertRaisesRegex(self, paramiko.ssh_exception.SSHException, expected_msg, mock.connect)