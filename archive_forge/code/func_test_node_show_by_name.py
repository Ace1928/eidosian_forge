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
def test_node_show_by_name(self):
    node = self.mgr.get(NODE1['name'])
    expect = [('GET', '/v1/nodes/%s' % NODE1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NODE1['uuid'], node.uuid)