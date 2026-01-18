import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
def test_is_port_not_trusted(self):
    self.assertFalse(net.is_port_trusted({'device_owner': constants.DEVICE_OWNER_COMPUTE_PREFIX + 'dev'}))