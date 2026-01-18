import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_invalid_string_to_dictionary(self):
    input_str = 'invalid'
    self.assertRaises(argparse.ArgumentTypeError, utils.str2dict, input_str)