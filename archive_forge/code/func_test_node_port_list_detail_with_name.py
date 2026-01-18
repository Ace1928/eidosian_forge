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
def test_node_port_list_detail_with_name(self):
    ports = self.mgr.list_ports(NODE1['name'], detail=True)
    expect = [('GET', '/v1/nodes/%s/ports/detail' % NODE1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(ports, HasLength(1))