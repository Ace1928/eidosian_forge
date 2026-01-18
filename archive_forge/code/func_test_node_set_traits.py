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
def test_node_set_traits(self):
    traits = ['CUSTOM_FOO', 'CUSTOM_BAR']
    resp = self.mgr.set_traits(NODE1['uuid'], traits)
    expect = [('PUT', '/v1/nodes/%s/traits' % NODE1['uuid'], {}, {'traits': traits})]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(resp)