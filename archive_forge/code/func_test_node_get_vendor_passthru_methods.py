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
def test_node_get_vendor_passthru_methods(self):
    vendor_methods = self.mgr.get_vendor_passthru_methods(NODE1['uuid'])
    expect = [('GET', '/v1/nodes/%s/vendor_passthru/methods' % NODE1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NODE_VENDOR_PASSTHRU_METHOD, vendor_methods)