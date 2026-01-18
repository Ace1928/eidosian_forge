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
def test_node_show_by_instance(self):
    node = self.mgr.get_by_instance_uuid(NODE2['instance_uuid'])
    expect = [('GET', '/v1/nodes/detail?instance_uuid=%s' % NODE2['instance_uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NODE2['uuid'], node.uuid)