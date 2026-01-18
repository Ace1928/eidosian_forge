import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_int_in_range(self):
    valid_inputs = [(1, -100, 100), ('1', -100, 100), (100, -100, 100), ('100', -100, 100), (-100, -100, 100), ('-100', -100, 100)]
    for input_value in valid_inputs:
        self.assertTrue(netutils._is_int_in_range(*input_value))