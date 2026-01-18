import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_key_files_and_key_material_arguments_are_mutual_exclusive(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_files': 'id_rsa', 'key_material': 'key'}
    expected_msg = 'key_files and key_material arguments are mutually exclusive'
    assertRaisesRegex(self, ValueError, expected_msg, ParamikoSSHClient, **conn_params)