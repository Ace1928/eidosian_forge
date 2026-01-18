import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_base_connection_key(self):
    ConnectionKey('foo')