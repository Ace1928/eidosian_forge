import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_get_interface_max_len(self):
    self.assertEqual(constants.DEVICE_NAME_MAX_LEN, len(utils.get_interface_name(LONG_NAME1)))
    self.assertEqual(10, len(utils.get_interface_name(LONG_NAME1, max_len=10)))
    self.assertEqual(12, len(utils.get_interface_name(LONG_NAME1, prefix='pre-', max_len=12)))