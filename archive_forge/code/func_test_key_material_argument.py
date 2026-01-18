import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
def test_key_material_argument(self):
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa.key')
    with open(path) as fp:
        private_key = fp.read()
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_material': private_key}
    mock = ParamikoSSHClient(**conn_params)
    mock.connect()
    pkey = paramiko.RSAKey.from_private_key(StringIO(private_key))
    expected_conn = {'username': 'ubuntu', 'allow_agent': False, 'hostname': 'dummy.host.org', 'look_for_keys': False, 'pkey': pkey, 'port': 22}
    mock.client.connect.assert_called_once_with(**expected_conn)
    self.assertLogMsg('Connecting to server')