import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_keep_alive_and_compression(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    mock_transport = Mock()
    client.client.get_transport = Mock(return_value=mock_transport)
    transport = client._get_transport()
    self.assertEqual(transport.set_keepalive.call_count, 0)
    self.assertEqual(transport.use_compression.call_count, 0)
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'keep_alive': 15, 'use_compression': True}
    client = ParamikoSSHClient(**conn_params)
    mock_transport = Mock()
    client.client.get_transport = Mock(return_value=mock_transport)
    transport = client._get_transport()
    self.assertEqual(transport.set_keepalive.call_count, 1)
    self.assertEqual(transport.use_compression.call_count, 1)