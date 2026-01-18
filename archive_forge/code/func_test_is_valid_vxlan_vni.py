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
def test_is_valid_vxlan_vni(self):
    for v in [constants.MIN_VXLAN_VNI, constants.MAX_VXLAN_VNI, constants.MIN_VXLAN_VNI + 1, constants.MAX_VXLAN_VNI - 1]:
        self.assertTrue(utils.is_valid_vxlan_vni(v))