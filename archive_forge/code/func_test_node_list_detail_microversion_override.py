import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_node_list_detail_microversion_override(self):
    nodes = self.mgr.list(detail=True, os_ironic_api_version='1.30')
    expect = [('GET', '/v1/nodes/detail', {'X-OpenStack-Ironic-API-Version': '1.30'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(nodes))
    self.assertEqual(nodes[0].extra, {})