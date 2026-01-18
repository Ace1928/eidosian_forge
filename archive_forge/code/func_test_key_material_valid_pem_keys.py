import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_key_material_valid_pem_keys(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa.key')
    with open(path) as fp:
        private_key = fp.read()
    pkey = client._get_pkey_object(key=private_key)
    self.assertTrue(pkey)
    self.assertTrue(isinstance(pkey, paramiko.RSAKey))
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_dsa.key')
    with open(path) as fp:
        private_key = fp.read()
    pkey = client._get_pkey_object(key=private_key)
    self.assertTrue(pkey)
    self.assertTrue(isinstance(pkey, paramiko.DSSKey))
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_ecdsa.key')
    with open(path) as fp:
        private_key = fp.read()
    pkey = client._get_pkey_object(key=private_key)
    self.assertTrue(pkey)
    self.assertTrue(isinstance(pkey, paramiko.ECDSAKey))