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
def test_node_set_provision_state_with_configdrive_invalid_bytes(self):
    invalid_utf8 = b'\xc3('
    target_state = 'active'
    self.assertRaisesRegex(ValueError, 'Config drive', self.mgr.set_provision_state, NODE1['uuid'], target_state, configdrive=invalid_utf8)