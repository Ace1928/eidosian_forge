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
def test_node_get_traits(self):
    traits = self.mgr.get_traits(NODE1['uuid'])
    expect = [('GET', '/v1/nodes/%s/traits' % NODE1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(TRAITS['traits'], traits)