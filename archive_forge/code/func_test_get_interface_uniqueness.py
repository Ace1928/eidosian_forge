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
def test_get_interface_uniqueness(self):
    prefix = 'prefix-'
    if_prefix1 = utils.get_interface_name(LONG_NAME1, prefix=prefix)
    if_prefix2 = utils.get_interface_name(LONG_NAME2, prefix=prefix)
    self.assertNotEqual(if_prefix1, if_prefix2)