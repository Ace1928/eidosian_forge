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
def test_node_volume_connector_list(self):
    volume_connectors = self.mgr.list_volume_connectors(NODE1['uuid'])
    expect = [('GET', '/v1/nodes/%s/volume/connectors' % NODE1['uuid'], {}, None)]
    self._validate_node_volume_connector_list(expect, volume_connectors)