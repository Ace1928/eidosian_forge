import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_consume_stderr_chunk_contains_part_of_multi_byte_utf8_character(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    client.CHUNK_SIZE = 1
    chan = Mock()
    chan.recv_stderr_ready.side_effect = [True, True, True, True, False]
    chan.recv_stderr.side_effect = ['ð', '\x90', '\x8d', '\x88']
    stderr = client._consume_stderr(chan).getvalue()
    self.assertEqual('ð\x90\x8d\x88', stderr)
    self.assertEqual(len(stderr), 4)