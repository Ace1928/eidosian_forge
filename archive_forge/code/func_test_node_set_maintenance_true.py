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
def test_node_set_maintenance_true(self):
    maintenance = self.mgr.set_maintenance(NODE1['uuid'], 'true', maint_reason='reason')
    body = {'reason': 'reason'}
    expect = [('PUT', '/v1/nodes/%s/maintenance' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(maintenance)