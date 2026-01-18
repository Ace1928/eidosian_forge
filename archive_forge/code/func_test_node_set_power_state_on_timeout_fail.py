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
def test_node_set_power_state_on_timeout_fail(self):
    self.assertRaises(ValueError, self.mgr.set_power_state, NODE1['uuid'], 'off', soft=False, timeout=0)