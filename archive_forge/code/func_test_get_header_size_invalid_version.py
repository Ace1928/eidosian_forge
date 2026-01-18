import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
def test_get_header_size_invalid_version(self):
    self.assertRaises(ValueError, zebra.ZebraMessage.get_header_size, 255)