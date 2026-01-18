import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_is_cidr(self):
    self.assertTrue(netutils.is_valid_cidr('10.10.10.0/24'))
    self.assertFalse(netutils.is_valid_cidr('10.10.10..0/24'))
    self.assertFalse(netutils.is_valid_cidr('wrong_cidr_format'))