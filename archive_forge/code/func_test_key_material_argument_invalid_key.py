import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
def test_key_material_argument_invalid_key(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_material': 'id_rsa'}
    mock = ParamikoSSHClient(**conn_params)
    expected_msg = 'Invalid or unsupported key type'
    assertRaisesRegex(self, paramiko.ssh_exception.SSHException, expected_msg, mock.connect)