import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
def test_destroy_node__node_not_found(self):
    """
        This simulates the case when destroy_node is being called for the 2nd
        time because some related resource failed to clean up, so the DELETE
        operation on the node will return 204 (because it was already deleted)
        but the method should return success.
        """

    def error(e, **kwargs):
        raise e(**kwargs)
    node = self.driver.list_nodes()[0]
    AzureMockHttp.responses = [lambda f: error(BaseHTTPError, code=204, message='No content')]
    ret = self.driver.destroy_node(node)
    self.assertTrue(ret)