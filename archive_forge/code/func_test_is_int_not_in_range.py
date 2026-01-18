import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_int_not_in_range(self):
    invalid_inputs = [(None, 1, 100), ('ten', 1, 100), (-1, 0, 255), ('None', 1, 100)]
    for input_value in invalid_inputs:
        self.assertFalse(netutils._is_int_in_range(*input_value))