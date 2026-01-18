import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_get_auth_password_but_given_ssh_key(self):
    n = NodeDriver('foo')
    n.features = {'create_node': ['password']}
    auth = NodeAuthSSHKey('publickey')
    self.assertRaises(LibcloudError, n._get_and_check_auth, auth)