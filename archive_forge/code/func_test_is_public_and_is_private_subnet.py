import sys
import random
import socket
import string
import os.path
import platform
import unittest
import warnings
from itertools import chain
import pytest
import libcloud.utils.files
from libcloud.utils.py3 import StringIO, b, bchr, urlquote, hexadigits
from libcloud.utils.misc import get_driver, set_driver, get_secure_random_string
from libcloud.common.types import LibcloudError
from libcloud.compute.types import Provider
from libcloud.utils.publickey import get_pubkey_ssh2_fingerprint, get_pubkey_openssh_fingerprint
from libcloud.utils.decorators import wrap_non_libcloud_exceptions
from libcloud.utils.networking import (
from libcloud.compute.providers import DRIVERS
from libcloud.compute.drivers.dummy import DummyNodeDriver
from libcloud.storage.drivers.dummy import DummyIterator
from io import FileIO as file
def test_is_public_and_is_private_subnet(self):
    public_ips = ['213.151.0.8', '86.87.86.1', '8.8.8.8', '8.8.4.4']
    private_ips = ['192.168.1.100', '10.0.0.1', '172.16.0.0']
    for address in public_ips:
        is_public = is_public_subnet(ip=address)
        is_private = is_private_subnet(ip=address)
        self.assertTrue(is_public)
        self.assertFalse(is_private)
    for address in private_ips:
        is_public = is_public_subnet(ip=address)
        is_private = is_private_subnet(ip=address)
        self.assertFalse(is_public)
        self.assertTrue(is_private)