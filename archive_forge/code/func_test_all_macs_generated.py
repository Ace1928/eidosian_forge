import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
def test_all_macs_generated(self):
    mac = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']
    generator = itertools.islice(net.random_mac_generator(mac), 70000)
    self.assertEqual(2 ** 16, len(list(generator)))