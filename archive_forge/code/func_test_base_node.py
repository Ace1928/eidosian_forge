import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_base_node(self):
    Node(id=0, name=0, state=0, public_ips=0, private_ips=0, driver=FakeDriver())