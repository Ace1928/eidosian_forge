import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
def test_deprecated_key_argument(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key': 'id_rsa'}
    mock = ParamikoSSHClient(**conn_params)
    mock.connect()
    expected_conn = {'username': 'ubuntu', 'allow_agent': False, 'hostname': 'dummy.host.org', 'look_for_keys': False, 'key_filename': 'id_rsa', 'port': 22}
    mock.client.connect.assert_called_once_with(**expected_conn)
    self.assertLogMsg('Connecting to server')