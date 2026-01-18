import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
def test_ed25519_key_type(self):
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_ed25519.key')
    with open(path) as fp:
        private_key = fp.read()
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_material': private_key}
    mock = ParamikoSSHClient(**conn_params)
    self.assertTrue(mock.connect())